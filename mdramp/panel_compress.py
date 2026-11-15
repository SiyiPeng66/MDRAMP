"""Score-independent, annotation-constrained protein-panel compression ladder."""
import argparse,json,hashlib,subprocess,shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def domains(entry):
    values=set()
    for x in entry.get('uniProtKBCrossReferences',[]):
        if x['database'] in ('Pfam','InterPro'):values.add(x['database']+':'+x['id'])
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='ecNumbers':values.update('EC:'+e['value'] for e in v)
                else:visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    visit(entry.get('proteinDescription',{}));return tuple(sorted(values))
def true(v):return str(v).lower()=='true'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel-dir','library-dir','output-dir','mmseqs'):p.add_argument('--'+name,required=True)
    p.add_argument('--alignment-cache');p.add_argument('--threads',type=int,default=4);a=p.parse_args();panel=Path(a.panel_dir);out=Path(a.output_dir)
    if out.exists():raise ValueError('Choose a new output version')
    frame=pd.read_csv(panel/'proteins.csv').fillna('');assert len(frame)==6413 and frame.record_id.is_unique and frame.sequence.is_unique
    members=pd.read_csv(panel/'panel_membership.csv').fillna('')
    library=pd.read_csv(panel/'library_evidence.csv').fillna('')
    wanted=set(members.member_id)|set(frame.record_id)
    evidence={};sources={}
    for file in Path(a.library_dir).rglob('*.json'):
        if 'raw' not in file.parts or file.name.endswith('.request.json'):continue
        data=json.loads(file.read_text())
        relevant=[r for r in data.get('results',[]) if r.get('primaryAccession') in wanted] if isinstance(data,dict) else []
        if relevant:sources[str(file.absolute())]=sha(file)
        for r in relevant:evidence[r['primaryAccession']]=domains(r)
    if wanted-set(evidence):raise ValueError('Missing raw annotation records')
    source_evidence=library.set_index('record_id')
    protected={};heterogeneous={};signatures={};source_mapping=[]
    for pid in frame.record_id:
        group=members[members.scoring_record_id==pid]
        originals=sorted(set(group.member_id)|{pid})
        entries=source_evidence.loc[originals]
        protected[pid]=entries.resistance_annotation.map(true).any()
        funcs={evidence[x] for x in originals}
        evidence_sets=set(entries.evidence_signature.astype(str))
        heterogeneous[pid]=len(funcs)!=1 or len(evidence_sets)!=1
        signatures[pid]=tuple(sorted(evidence_sets))
        evidence[pid]=tuple(sorted({v for x in funcs for v in x}))
        for original in originals:source_mapping.append({'scoring_record_id':pid,'original_record_id':original,'resistance_annotation':true(source_evidence.loc[original].resistance_annotation)})
    frame['resistance_annotation']=frame.record_id.map(protected)
    frame['source_annotation_heterogeneous']=frame.record_id.map(heterogeneous)
    out.mkdir(parents=True)
    inputs={str((panel/'proteins.csv').absolute()):sha(panel/'proteins.csv'),str(Path(a.mmseqs).absolute()):sha(a.mmseqs),str(Path(__file__).absolute()):sha(__file__),**sources,str((panel/'panel_membership.csv').absolute()):sha(panel/'panel_membership.csv'),str((panel/'library_evidence.csv').absolute()):sha(panel/'library_evidence.csv')}
    rules={'baseline_count':6413,'identity_thresholds':[.9,.7,.5],'identity_definition':'exact aligned identities / max(full query length,full target length)','bidirectional_coverage_min':.9,'protected':'every explicit resistance_annotation protein retained as itself','strata':['represented_species','evidence_signature'],'source_policy':'same-species cross-source merging allowed; all original member annotations aggregated, heterogeneous evidence/functions cannot merge; no phenotype inference','function_constraint':'identical nonempty Pfam+InterPro+EC ID sets; all-empty annotations cannot merge','representative_order':'reviewed desc, annotation_score desc, length desc, accession ascending','mic_or_peptide_scores_used':False,'interpretation':'sequence-family proxy only; low identity does not establish functional equivalence','input_sha256':inputs}
    (out/'rules.json').write_text(json.dumps(rules,indent=2))
    fasta=out/'baseline.fasta';fasta.write_text(''.join(f'>{r.record_id}\n{r.sequence}\n' for r in frame.itertuples()))
    cmd=[a.mmseqs,'easy-search',str(fasta),str(fasta),str(out/'alignments.tsv'),str(out/'tmp'),'--min-seq-id','0.45','-c','0.9','--cov-mode','0','--alignment-mode','3','-a','--max-seqs','10000','-s','7.5','--threads',str(a.threads),'--format-output','query,target,nident,qlen,tlen,qcov,tcov']
    if a.alignment_cache:
        cache=Path(a.alignment_cache);old=json.loads((cache/'rules.json').read_text())
        if old['input_sha256'].get(str((panel/'proteins.csv').absolute()))!=sha(panel/'proteins.csv'):raise ValueError('Alignment cache baseline differs')
        manifest=json.loads((cache/'manifest.json').read_text())
        if manifest['output_sha256']['alignments.tsv']!=sha(cache/'alignments.tsv'):raise ValueError('Alignment cache changed')
        shutil.copyfile(cache/'alignments.tsv',out/'alignments.tsv')
        (out/'alignment_cache.json').write_text(json.dumps({'source':str(cache.absolute()),'sha256':sha(cache/'alignments.tsv')}))
    else:
        with (out/'mmseqs.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    pd.DataFrame(source_mapping).to_csv(out/'original_member_evidence.csv',index=False)
    hits=pd.read_csv(out/'alignments.tsv',sep='\t',names=['query','target','nident','qlen','tlen','qcov','tcov']);hits['global_identity']=hits.nident/hits[['qlen','tlen']].max(axis=1)
    self_hits=hits[hits['query']==hits['target']]
    if len(self_hits)!=len(frame) or not self_hits.global_identity.eq(1).all():raise ValueError('MMseqs exact identity requires backtrace; self identities must equal one')
    links={}
    for r in hits.itertuples():
        if min(r.qcov,r.tcov)>=.9:links[(r.query,r.target)]=max(links.get((r.query,r.target),0),r.global_identity)
    table=frame.set_index('record_id');order=frame.assign(_review=frame.reviewed.map(true)).sort_values(['_review','annotation_score','length','record_id'],ascending=[False,False,False,True]).record_id.tolist()
    def signature(pid):
        r=table.loc[pid];return (str(r.represented_species),signatures[pid],evidence[pid])
    summary=[]
    pd.DataFrame([{'record_id':k,'annotation_ids':';'.join(v)} for k,v in evidence.items()]).to_csv(out/'functional_signatures.csv',index=False)
    for threshold in rules['identity_thresholds']:
        kept=[];mapping=[];groups=defaultdict(list)
        for pid in order:
            r=table.loc[pid];representative=pid;identity=1.;sig=signature(pid)
            if not true(r.resistance_annotation) and evidence[pid] and not heterogeneous[pid]:
                for candidate in groups[sig]:
                    score=max(links.get((pid,candidate),0),links.get((candidate,pid),0))
                    if score>=threshold:representative=candidate;identity=score;break
            if representative==pid:kept.append(pid)
            if representative==pid and not heterogeneous[pid]:groups[sig].append(pid)
            mapping.append({'member_id':pid,'representative_id':representative,'global_identity':identity,'protected_resistance':true(r.resistance_annotation)})
        directory=out/f'identity_{int(threshold*100)}';directory.mkdir();selected=frame[frame.record_id.isin(kept)]
        selected.to_csv(directory/'proteins.csv',index=False);pd.DataFrame(mapping).to_csv(directory/'membership.csv',index=False)
        (directory/'targets.fasta').write_text(''.join(f'>{r.record_id}\n{r.sequence}\n' for r in selected.itertuples()))
        coverage=[]
        for column in rules['strata']+['represented_assays','gene_names']:
            before=set(frame[column].astype(str));after=set(selected[column].astype(str));coverage.append({'dimension':column,'before_unique':len(before),'after_unique':len(after),'lost_values':sorted(before-after)})
        (directory/'coverage.json').write_text(json.dumps(coverage,indent=2));summary.append({'identity':threshold,'proteins':len(selected),'removed':len(frame)-len(selected),'protected_resistance_retained':int(selected.resistance_annotation.map(true).sum())})
    if any(sha(k)!=v for k,v in inputs.items()):raise ValueError('Inputs changed')
    pd.DataFrame(summary).to_csv(out/'compression_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'rules_sha256':sha(out/'rules.json'),'status':'complete','summary':summary,'mmseqs_command':cmd,'mmseqs_version':subprocess.check_output([a.mmseqs,'version'],text=True).strip(),'output_sha256':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and 'tmp' not in f.relative_to(out).parts}},indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
