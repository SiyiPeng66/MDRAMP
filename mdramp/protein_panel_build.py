"""Build an auditable annotation/evidence-supported prospective protein panel."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def fasta(path, rows, idcol='record_id'):
    Path(path).write_text(''.join(f'>{r[idcol]}\n{r["sequence"]}\n' for r in rows))


def normalize(entry, species, proteome):
    sequence=entry.get('sequence',{}).get('value','')
    reasons=[]
    if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',sequence):reasons.append('nonstandard_or_empty_sequence')
    if entry.get('proteinDescription',{}).get('flag')=='Fragment':reasons.append('fragment')
    cautions=[c for c in entry.get('comments',[]) if c.get('commentType')=='SEQUENCE CAUTION']
    if any(v.get('type')=='Erroneous gene model prediction' for c in cautions for v in c.get('sequenceCautions',[])):
        reasons.append('erroneous_gene_model')
    description=entry.get('proteinDescription',{})
    name=description.get('recommendedName',{}).get('fullName',{}).get('value','')
    if not name:name=next((v.get('fullName',{}).get('value','') for v in description.get('submissionNames',[])), '')
    if re.search(r'\bpseudogene\b',name,re.I):reasons.append('pseudogene_in_protein_name')
    return dict(record_id=entry['primaryAccession'],species=species,proteome_id=proteome,
        organism=entry['organism']['scientificName'],taxonomy_id=entry['organism']['taxonId'],
        gene_names=';'.join(g.get('geneName',{}).get('value','') for g in entry.get('genes',[])),
        protein_name=name,sequence=sequence,length=len(sequence),sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
        reviewed=entry.get('entryType')=='UniProtKB reviewed (Swiss-Prot)',
        annotation_score=entry.get('annotationScore',0),keyword_ids=';'.join(k['id'] for k in entry.get('keywords',[])),
        keyword_names=';'.join(k['name'] for k in entry.get('keywords',[])),
        keyword_evidence_json=json.dumps(entry.get('keywords',[]),ensure_ascii=False),
        location_evidence_json=json.dumps([c for c in entry.get('comments',[]) if c.get('commentType')=='SUBCELLULAR LOCATION'],ensure_ascii=False),
        quality_eligible=not reasons,quality_exclusion=';'.join(reasons),
        long_sequence_review_required=len(sequence)>2048,source_url='https://www.uniprot.org/uniprotkb/'+entry['primaryAccession'])


def annotate(row, rules):
    ids=set(row['keyword_ids'].split(';'))
    foreign=ids&{k['id'] for k in rules['non_bacterial_location_keywords']}
    row['non_bacterial_location_keywords']=';'.join(sorted(foreign))
    for kind in ('envelope','resistance'):
        accepted={k['id'] for k in rules['keyword_closure'][kind]}
        if kind=='envelope' and foreign:
            accepted-=set(rules['generic_membrane_keywords_suppressed_when_foreign_location_present'])
        matches=sorted(ids&accepted)
        row[kind+'_annotation']=bool(matches)
        row[kind+'_matched_keywords']=';'.join(matches)
    return row


def execute(command, log):
    with Path(log).open('w') as handle:
        subprocess.run([str(x) for x in command], stdout=handle, stderr=subprocess.STDOUT, check=True)


def essential_matches(frame, records, out, binary, threads):
    """Same-species sequence mapping; no gene-name substitution or label transfer."""
    matches=[]
    for species, group in frame[frame.quality_eligible].groupby('species',sort=True):
        source=records[records.species.eq(species)].copy()
        source=source[source.sequence.fillna('').str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+')]
        if source.empty:continue
        folder=out/species;folder.mkdir(parents=True)
        fasta(folder/'reference.fasta',group.to_dict('records'))
        fasta(folder/'essential.fasta',source.to_dict('records'))
        argv=[binary,'easy-search',folder/'reference.fasta',folder/'essential.fasta',folder/'matches.tsv',folder/'tmp',
            '--min-seq-id','0.9','-c','0.9','--cov-mode','0','--alignment-mode','3','--seq-id-mode','0',
            '-e','1e-5','--max-seqs','10000','--threads',str(threads),
            '--format-output','query,target,fident,qcov,tcov,evalue,bits']
        execute(argv,folder/'search.log');dump(folder/'command.json',[str(v) for v in argv])
        # Exact sequence evidence is enumerated independently of heuristic search.
        exact={}
        for r in source.to_dict('records'):exact.setdefault(r['sequence'],[]).append(r['record_id'])
        source_seq=dict(zip(source.record_id,source.sequence));ref_seq=dict(zip(group.record_id,group.sequence))
        found={}
        for r in group.to_dict('records'):
            for sid in exact.get(r['sequence'],[]):found[(r['record_id'],sid)]=dict(identity=1.,query_coverage=1.,target_coverage=1.,evalue=0.,mapping='exact_sequence')
        for line in (folder/'matches.tsv').read_text().splitlines():
            q,t,identity,qcov,tcov,evalue,bits=line.split('\t')
            if q not in ref_seq or t not in source_seq:raise ValueError('Unknown alignment ID')
            values=list(map(float,(identity,qcov,tcov,evalue)))
            if not all(math.isfinite(v) for v in values) or not all(0<=v<=1 for v in values[:3]) or values[3]<0:raise ValueError('Invalid essentiality alignment values')
            if values[0]<.9 or values[1]<.9 or values[2]<.9 or values[3]>1e-5:raise ValueError('Out-of-rule essentiality match')
            if (q,t) not in found:found[q,t]=dict(identity=values[0],query_coverage=values[1],target_coverage=values[2],evalue=values[3],mapping='same_species_homology_inference')
        for (q,t),details in sorted(found.items()):matches.append(dict(record_id=q,essential_record_id=t,species=species,**details))
        print(f'{species}: essentiality mappings {len(found)}',flush=True)
    columns=['record_id','essential_record_id','species','identity','query_coverage','target_coverage','evalue','mapping']
    result=pd.DataFrame(matches,columns=columns);result.to_csv(out/'essential_mappings.csv',index=False)
    return result


def cluster(frame, out, binary, threshold, threads):
    out.mkdir(parents=True);members=[]
    work=frame.copy()
    work['source_stratum']=work['proteome_id'] if 'proteome_id' in work else 'single_source'
    for (species,source_stratum,signature),group in work.groupby(['species','source_stratum','evidence_signature'],sort=True):
        folder=out/(species+'_'+source_stratum.replace(';','_')+'_'+signature);folder.mkdir()
        group=group.sort_values('record_id');fasta(folder/'input.fasta',group.to_dict('records'))
        if len(group)==1:pairs=[(group.iloc[0].record_id,group.iloc[0].record_id)]
        else:
            argv=[binary,'easy-cluster',folder/'input.fasta',folder/'result',folder/'tmp',
                  '--min-seq-id',str(threshold),'-c','0.9','--cov-mode','0','--seq-id-mode','0',
                  '--alignment-mode','3','--cluster-mode','2','--single-step-clustering','1',
                  '--threads',str(threads),'--max-seqs','10000']
            execute(argv,folder/'cluster.log');dump(folder/'command.json',[str(v) for v in argv])
            pairs=[tuple(line.split('\t')) for line in (folder/'result_cluster.tsv').read_text().splitlines()]
        ids=set(group.record_id);seen=[m for _,m in pairs]
        if len(seen)!=len(set(seen)) or set(seen)!=ids or not {r for r,_ in pairs}<=ids:raise ValueError('Invalid cluster coverage')
        pair_set=set(pairs)
        if not all((r,r) in pair_set for r,_ in pairs):raise ValueError('Cluster representatives must belong to their own clusters')
        members.extend(dict(species=species,source_stratum=source_stratum,evidence_signature=signature,representative_id=r,member_id=m,identity_threshold=threshold) for r,m in pairs)
    result=pd.DataFrame(members);result.to_csv(out/'memberships.csv',index=False);return result


def build(a):
    root=Path(a.library_dir).absolute();out=Path(a.output_dir).absolute();rules=json.loads(Path(a.rules).read_text())
    library_manifest=Path(getattr(a,'library_manifest',None) or root/'download_manifest.json').absolute()
    manifest=json.loads(library_manifest.read_text())
    if manifest.get('status')!='complete':raise ValueError('Complete source library required')
    for name,digest in manifest['output_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Source library hash mismatch')
    if sha(root/'rules_sources/uniprot_keywords.json')!=rules['keyword_source_sha256']:raise ValueError('Keyword ontology changed')
    essential_manifest_path=Path(a.essential_manifest).absolute()
    essential_provenance=json.loads(essential_manifest_path.read_text())
    if essential_provenance.get('status')!='official_snapshot_parsed':raise ValueError('Official parsed DEG snapshot required')
    if essential_provenance.get('outputs_sha256',{}).get(Path(a.essential_records).name)!=sha(a.essential_records):raise ValueError('Essential records provenance mismatch')
    for name,digest in essential_provenance['inputs_sha256'].items():
        if sha(essential_manifest_path.parent/name)!=digest:raise ValueError('DEG source changed')
    if out.exists():raise ValueError('Never overwrite a panel output')
    out.mkdir(parents=True)
    if manifest.get('assay_source_mapping'):
        pd.DataFrame(manifest['assay_source_mapping']).to_csv(out/'assay_source_mapping.csv',index=False)
    inputs={str(Path(p).absolute()):sha(p) for p in [a.rules,a.essential_records,a.essential_manifest,a.mmseqs,__file__,library_manifest]}
    inputs[str(root/'rules_sources/uniprot_keywords.json')]=rules['keyword_source_sha256']
    inputs.update({str(root/name):digest for name,digest in manifest['output_sha256'].items()})
    inputs.update({str(essential_manifest_path.parent/name):digest for name,digest in essential_provenance['inputs_sha256'].items()})
    rows=[]
    for library_key,info in manifest['chosen_proteomes'].items():
        species=info.get('species',library_key)
        entries=[]
        paths=[root/p for p in info['protein_json_files']] if 'protein_json_files' in info else sorted((root/'raw').glob(f'{species}_{info["proteome_id"]}_proteins_*.json'))
        bound_paths={(root/p).resolve() for p in manifest['output_sha256']}
        for path in paths:
            if path.name.endswith('.request.json'):continue
            if path.resolve() not in bound_paths:raise ValueError('Protein JSON source is not hash-bound by library manifest')
            entries.extend(json.loads(path.read_text())['results'])
        if len(entries)!=info['downloaded_protein_count']:raise ValueError('Library count mismatch')
        for e in entries:
            row=annotate(normalize(e,species,info['proteome_id']),rules)
            row['source_assays']=';'.join(info.get('assays',[]));rows.append(row)
    frame=pd.DataFrame(rows)
    source_occurrences=len(frame)
    frame[['record_id','species','proteome_id','organism','source_assays']].to_csv(out/'library_source_membership.csv',index=False)
    if frame.record_id.duplicated().any():
        for _,g in frame.groupby('record_id'):
            if g.sequence.nunique()!=1 or g.species.nunique()!=1:raise ValueError('Conflicting repeated library accession')
            if any(g[c].astype(str).nunique()!=1 for c in frame.columns if c not in {'proteome_id','source_assays'}):raise ValueError('Conflicting annotations for repeated library accession')
        sources=frame.groupby('record_id').proteome_id.apply(lambda v:';'.join(sorted(set(v))))
        assays=frame.groupby('record_id').source_assays.apply(lambda v:';'.join(sorted({x for y in v for x in y.split(';') if x})))
        frame=frame.drop_duplicates('record_id').copy();frame['proteome_id']=frame.record_id.map(sources);frame['source_assays']=frame.record_id.map(assays)
    records=pd.read_csv(a.essential_records,keep_default_na=False)
    if not {'record_id','species','sequence'}.issubset(records) or records.record_id.duplicated().any():raise ValueError('Unique essential record IDs required')
    records.to_csv(out/'essential_source_records.csv',index=False)
    dump(out/'essential_source_provenance.json',essential_provenance)
    mapping_dir=out/'essential_mapping';mapping_dir.mkdir()
    mapping=essential_matches(frame,records,mapping_dir,a.mmseqs,a.threads)
    exact=set(mapping.loc[mapping.mapping.eq('exact_sequence'),'record_id'])
    inferred=set(mapping.loc[mapping.mapping.eq('same_species_homology_inference'),'record_id'])
    frame['essential_exact_sequence']=frame.record_id.isin(exact)
    frame['essential_homology_only']=frame.record_id.isin(inferred-exact)
    frame['essential_support']=frame.essential_exact_sequence|frame.essential_homology_only
    frame['evidence_signature']=frame.apply(lambda r:''.join(str(int(r[k])) for k in ['envelope_annotation','resistance_annotation','essential_support']),axis=1)
    frame['selected_by_evidence']=frame.quality_eligible&frame.evidence_signature.ne('000')
    frame['selection_reason']=frame.apply(lambda r:';'.join(k for k in ['envelope_annotation','resistance_annotation','essential_exact_sequence','essential_homology_only'] if r[k]),axis=1)
    frame['exclusion_reason']=frame.apply(lambda r:r.quality_exclusion or ('no_prespecified_evidence' if not r.selected_by_evidence else ''),axis=1)
    frame.to_csv(out/'library_evidence.csv',index=False)
    frame[~frame.selected_by_evidence].to_csv(out/'excluded_records.csv',index=False)
    selected=frame[frame.selected_by_evidence].copy();selected.to_csv(out/'evidence_selected_records.csv',index=False)
    if selected.empty:raise ValueError('No evidence-supported candidates')
    memberships=cluster(selected,out/'clusters_90',a.mmseqs,.9,a.threads)
    representatives=set(memberships.representative_id);reps=selected[selected.record_id.isin(representatives)].sort_values(['species','record_id'])
    canonical={};rep_to_scoring={}
    for r in reps.to_dict('records'):
        canonical.setdefault(r['sequence'],r['record_id']);rep_to_scoring[r['record_id']]=canonical[r['sequence']]
    memberships['scoring_record_id']=memberships.representative_id.map(rep_to_scoring)
    memberships['source_assays']=memberships.member_id.map(frame.set_index('record_id').source_assays)
    memberships.to_csv(out/'panel_membership.csv',index=False)
    panel=reps.drop_duplicates('sequence').copy()
    panel['represented_species']=panel.record_id.map(memberships.groupby('scoring_record_id').species.apply(lambda x:';'.join(sorted(set(x)))))
    panel['represented_source_records']=panel.record_id.map(memberships.groupby('scoring_record_id').size())
    panel['represented_assays']=panel.record_id.map(memberships.groupby('scoring_record_id').source_assays.apply(lambda v:';'.join(sorted({x for y in v for x in y.split(';') if x}))))
    panel.to_csv(out/'proteins.csv',index=False);fasta(out/'targets.fasta',panel.to_dict('records'))
    fasta(out/'library_clean.fasta',frame[frame.quality_eligible].to_dict('records'))
    counts=[]
    for species,g in frame.groupby('species',sort=True):
        counts.append(dict(species=species,raw_records=len(g),clean_records=int(g.quality_eligible.sum()),
            envelope_annotation=int((g.quality_eligible&g.envelope_annotation).sum()),resistance_annotation=int((g.quality_eligible&g.resistance_annotation).sum()),
            essential_exact_sequence=int((g.quality_eligible&g.essential_exact_sequence).sum()),essential_homology_only=int((g.quality_eligible&g.essential_homology_only).sum()),
            evidence_union=int(g.selected_by_evidence.sum()),exact_unique_selected=g[g.selected_by_evidence].sequence.nunique(),
            cluster_representatives=int(reps.species.eq(species).sum()),scoring_sequences_with_species=memberships.loc[memberships.species.eq(species),'scoring_record_id'].nunique()))
    pd.DataFrame(counts).to_csv(out/'counts_by_species.csv',index=False)
    frame[frame.selected_by_evidence].groupby(['species','evidence_signature']).size().rename('count').reset_index().to_csv(out/'evidence_intersections.csv',index=False)
    sensitivity=[]
    for threshold in rules['sensitivity_cluster_identity']:
        m=cluster(selected,out/f'clusters_{round(threshold*100)}',a.mmseqs,threshold,a.threads)
        for species,g in m.groupby('species'):sensitivity.append(dict(identity=threshold,species=species,representatives=g.representative_id.nunique()))
    pd.DataFrame(sensitivity).to_csv(out/'cluster_size_sensitivity.csv',index=False)
    # Quantify residual between-stratum homology without changing the main panel.
    audit=panel.copy();audit['species']='all';audit['evidence_signature']='all';audit['proteome_id']='all'
    global_members=cluster(audit,out/'cross_stratum_redundancy_audit',a.mmseqs,.9,a.threads)
    cross_stratum_count=len(panel)-global_members.representative_id.nunique()
    labels={'Ab':'鲍曼不动杆菌','Sa':'金黄色葡萄球菌','Kp':'肺炎克雷伯菌','Pa':'铜绿假单胞菌'}
    lines=['# 本次实验参考蛋白库与评分面板', '',
        f'来源：UniProt {manifest["uniprot_release"]}；{len(manifest["chosen_proteomes"])}个版本化蛋白组。公共菌株标识匹配或参考代理不等同于实际实验样品基因组确认。', '',
        '|物种|原始蛋白|质量合格|证据并集入选|分层聚类代表|',
        '|---|---:|---:|---:|---:|']
    for r in counts:lines.append(f'|{labels[r["species"]]}|{r["raw_records"]}|{r["clean_records"]}|{r["evidence_union"]}|{r["cluster_representatives"]}|')
    lines += ['',f'入选记录共 {len(selected)} 条；物种、来源蛋白组及证据类别内90%一致性、90%双向覆盖聚类后共 {len(reps)} 个代表；再合并完全相同的序列，最终评分面板为 **{len(panel)} 条独立蛋白序列**。',
        '', '各证据类别可以重叠，不能相加得到并集数量。跨层仍可能有近同源序列；全局90%审计进一步减少的代表数为 '+str(cross_stratum_count)+'，因此不称本面板为全局90%非冗余集合。',
        '', '## 文件与用途', '',
        '- `library_evidence.csv`：完整库、序列、注释、质量判定和入选理由。',
        '- `assay_source_mapping.csv`（若提供）：六类实验菌的公共菌株标识匹配、实际采用的来源和回退原因。',
        '- `library_clean.fasta`：质量合格的完整候选库。',
        '- `proteins.csv` / `targets.fasta`：本次构建的评分面板序列。',
        '- `evidence_selected_records.csv`：去冗余前所有符合证据规则的记录。',
        '- `panel_membership.csv`：原始入选记录→分层聚类代表→实际评分序列的映射。',
        '- `excluded_records.csv`：质量不合格或未命中预设证据规则的记录及原因。',
        '- `counts_by_species.csv` / `evidence_intersections.csv`：分物种数量与证据交集。',
        '- `essential_mapping/essential_mappings.csv` / `essential_source_records.csv`：必需性映射、原始研究及条件。',
        '- `cluster_size_sensitivity.csv`：80%和95%聚类阈值下的数量；不是预测排名稳定性验证。',
        '- `manifest.json`：来源版本、参数、代码及文件哈希。',
        '', '## 证据与限制', '',
        '包膜与耐药类别来自UniProt受控关键词及经审查的层级关系，包含自动注释；不是AMP结合实验证据，也没有冒称完成CARD映射。',
        '必需性同序列匹配仅表示与DEG来源实验记录的蛋白序列相同；同物种90%一致性、90%双向覆盖的非完全匹配明确标为推断。必需性取决于原研究菌株和条件。',
        '此次DEG官方快照缺少肺炎克雷伯菌记录，不能解释为该菌没有必需基因，也未用其他物种替补。来源中缺失及异常序列保留记录。',
        '本次未使用230条肽的MIC、肽模型分数或湿实验结果选择蛋白，不限制入选总量。旧16蛋白面板不作为本次最终面板。',
        '这是新建前瞻参考面板，不能追溯宣称R0前已经冻结。面板构建已完成不代表ESM编码、最终MIC校准、预测排序稳定性或新湿实验验证已完成。',
        '全局top-5互作聚合不直接证明跨物种或广谱抗菌作用；后续仍须检查面板组成及规模对候选排序的影响。',
        '', '## 复现入口', '', '```bash',
        'python main.py prepare protein-panel --config configs/prospective/protein_panel_v2.yaml', '```',
        '', '命令拒绝覆盖已有面板；复算时需使用新的输出目录。']
    (out/'本次蛋白库与面板说明.md').write_text('\n'.join(lines)+'\n')
    if any(sha(p)!=digest for p,digest in inputs.items()):raise ValueError('Build input changed')
    result=dict(status='constructed_pending_scoring_validation',created_at=datetime.now(timezone.utc).isoformat(),
        purpose=rules['purpose'],actual_assay_isolate_match=False,historical_r0_panel=False,
        source_record_occurrences_count=source_occurrences,
        raw_count=len(frame),clean_count=int(frame.quality_eligible.sum()),evidence_selected_count=len(selected),
        cluster_representative_count=len(reps),scoring_protein_count=len(panel),counts_by_species=counts,
        essential_source_record_count=len(records),essential_species_available=sorted(set(records.species)),
        essential_source_usable_sequence_count=int(records.sequence.fillna('').str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+').sum()),
        residual_global_90pct_redundant_representatives=cross_stratum_count,
        resistance_evidence='UniProt keyword annotations; CARD mapping not performed',
        no_mic_or_peptide_score_used=True,input_sha256=inputs,rules=rules,source_manifest=manifest['chosen_proteomes'],
        uniprot_release=manifest['uniprot_release'],mmseqs_version=subprocess.check_output([a.mmseqs,'version'],text=True).strip(),
        assay_source_mapping=manifest.get('assay_source_mapping',[]),
        output_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'tmp' not in p.parts})
    dump(out/'manifest.json',result);print(json.dumps({'raw':len(frame),'selected':len(selected),'scoring_proteins':len(panel),'by_species':counts}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['library-dir','rules','essential-records','essential-manifest','mmseqs','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--library-manifest')
    p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    if a.threads<1:raise ValueError('Positive threads required')
    build(a)


if __name__=='__main__':main()
