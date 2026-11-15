"""Prespecified mechanism-focused and essential-inclusive panel ladders."""
import argparse,json,re,os,tempfile,shutil
from pathlib import Path
import pandas as pd
from src.utils.hash import sha256_file
from mdramp.panel_compress import domains,true
PATTERNS={
'cell_wall_envelope':r'\b(peptidoglycan|murein|penicillin.binding|cell.wall|lipopolysaccharide|lipooligosaccharide|lipid A|teichoic|undecaprenyl|outer.membrane|porin|lipoprotein.maturation|lipoprotein.signal.peptidase)\b',
'membrane_lipid_homeostasis':r'\b(phospholipid|cardiolipin|phosphatidylglycerol|phosphatidylethanolamine|phosphatidylserine|phosphatidylcholine|phosphatidylinositol|lipid.transport|lipid.flippase|membrane.protease|membrane.protein.insertase)\b',
'envelope_assembly':r'\b(beta.barrel.assembly|lipopolysaccharide.assembly|lipopolysaccharide.transport|protein.translocase|protein.export|signal.peptidase|periplasmic.chaperone)\b',
'ion_homeostasis':r'\b(proton|sodium|potassium|calcium|magnesium|chloride)\b.*\b(antiporter|transporter|channel|pump)\b',
'explicit_efflux':r'\b(efflux|multidrug.transporter|drug.transporter)\b',
'functional_transport':r'\b(transporter|permease|symporter|antiporter|channel|transport.protein|transport.system|translocase)\b'}
UNKNOWN=re.compile(r'\b(uncharacterized|hypothetical|unknown.function|DUF[ -]?\d+|UPF[ -]?\d+)\b',re.I)

def normalized_name(name):
    name=re.sub(r'\b(putative|probable|possible|predicted)\b','',name.lower())
    return re.sub(r'\s+',' ',name).strip(' -')

def classify(name,envelope,essential,resistance):
    groups=[k for k,p in PATTERNS.items() if re.search(p,name,re.I)]
    defined=bool(name.strip()) and not UNKNOWN.search(name) and normalized_name(name) not in {'protein','membrane protein','integral membrane protein','transmembrane protein','conserved protein','hypothetical protein'}
    if resistance:groups.append('resistance_annotation')
    if essential and defined:groups.append('functionally_named_essential')
    if 'functional_transport' in groups and not envelope:groups.remove('functional_transport')
    return sorted(set(groups)),bool(defined)

def family(entry):
    annotation=domains(entry);pfam=tuple(x for x in annotation if x.startswith('Pfam:'));ec=tuple(x for x in annotation if x.startswith('EC:'))
    ordered=tuple(x.get('description','') for x in sorted([x for x in entry.get('features',[]) if x.get('type')=='Domain' and x.get('location',{}).get('start',{}).get('value') is not None],key=lambda x:x['location']['start']['value']))
    location=tuple(sorted({x.get('location',{}).get('value','') for c in entry.get('comments',[]) if c.get('commentType')=='SUBCELLULAR LOCATION' for x in c.get('subcellularLocations',[])}))
    return pfam,ec,ordered,location

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel-dir','library-dir','compression-dir','output-dir'):p.add_argument('--'+name,required=True)
    a=p.parse_args();panel=Path(a.panel_dir).resolve();library=Path(a.library_dir).resolve();compression=Path(a.compression_dir).resolve();out=Path(a.output_dir).resolve()
    if out.exists():raise ValueError('Output already exists')
    files=[panel/'proteins.csv',panel/'panel_membership.csv',panel/'library_evidence.csv',compression/'alignments.tsv',compression/'manifest.json',Path(__file__).resolve()]
    inputs={str(x):sha256_file(x) for x in files};cm=json.loads((compression/'manifest.json').read_text())
    if cm['output_sha256'].get('alignments.tsv')!=sha256_file(compression/'alignments.tsv'):raise ValueError('Alignment integrity mismatch')
    frame=pd.read_csv(panel/'proteins.csv').fillna('');membership=pd.read_csv(panel/'panel_membership.csv');ev=pd.read_csv(panel/'library_evidence.csv',low_memory=False).fillna('').set_index('record_id')
    wanted=set(membership.member_id)|set(frame.record_id);raw={}
    for path in library.rglob('*.json'):
        if 'raw' not in path.parts or path.name.endswith('.request.json'):continue
        content=json.loads(path.read_text());records=[x for x in content.get('results',[]) if x.get('primaryAccession') in wanted] if isinstance(content,dict) else []
        if records:inputs[str(path)]=sha256_file(path)
        raw.update({x['primaryAccession']:x for x in records})
    if wanted-set(raw):raise ValueError('Incomplete original annotation evidence')
    rules={'patterns':PATTERNS,'matching':'case insensitive word-boundary matching of source protein_name only; unknown/hypothetical/DUF excluded from functionally_named_essential; functional_transport also needs envelope annotation','tiers':['core','core_transport','core_transport_essential'],'scope':'core is a mechanism-focused prospective scope change, not equivalent retention of the 6413 panel. Third tier restores functionally named essential-supported intracellular proteins.','protected':'all any-original-member resistance_annotation targets retained, annotation is not proof of variant resistance or peptide binding','redundancy':'same represented species, same functional group set, compatible identical Pfam composition/EC/ordered UniProt Domain descriptions/subcellular location; heterogeneity among original members prevents merge; direct full-length identity90 or50 and bidirectional coverage90 required','name_based_scope_caution':'Protein-name rules are incomplete functional proxies; not exhaustive GO reasoning, experimental validation, or proof of essentiality. Source group heterogeneity is reported.','architecture_caution':'Pfam IDs without coordinates are composition, not ordered domain architecture. Ordered Domain features used only where directly provided.','representative_tie':'reviewed then annotation score then length descending, then ID ascending','specific_function_constraint':'Group redundancy also requires identical normalized specific original protein names (qualifiers putative/probable/possible/predicted removed); generic membrane/protein names cannot establish named essential function.','evidence_caution':'reviewed_annotation is curation status, not experimental support; full source keyword/location evidence retained; ECO/GO semantic reasoning not performed.','no_mic_or_peptide_scores':True,'target_count_not_fixed':True}
    out.mkdir(parents=True);(out/'selection_rules.json').write_text(json.dumps(rules,indent=2))
    details=[];fam={};groupmap={};all_domains={};names={};protected=set();heterogeneous=set()
    for r in frame.itertuples():
        ids=sorted(set(membership.loc[membership.scoring_record_id==r.record_id,'member_id'])|{r.record_id});groups=set();fs=set();evidence=[];source_groups=set();source_names=set();all_domains[r.record_id]=set()
        for original in ids:
            row=ev.loc[original];source_names.add(normalized_name(str(row.protein_name)));res=true(row.resistance_annotation);essential=true(row.essential_support);env=true(row.envelope_annotation)
            g,defined=classify(str(row.protein_name),env,essential,res);groups.update(g);source_groups.add(tuple(g));fs.add(family(raw[original]));all_domains[r.record_id].update(domains(raw[original]))
            if res:protected.add(r.record_id)
            evidence.append({'record_id':original,'protein_name':str(row.protein_name),'groups':g,'reviewed':true(row.reviewed),'essential_exact':true(row.essential_exact_sequence),'essential_homology_only':true(row.essential_homology_only),'resistance_annotation':res,'annotation_evidence_level':'reviewed_annotation' if true(row.reviewed) else 'automatic_annotation','source_url':str(row.source_url),'keyword_evidence_json':str(row.keyword_evidence_json),'location_evidence_json':str(row.location_evidence_json)})
        names[r.record_id]=tuple(sorted(source_names))
        if len(fs)!=1 or len(source_groups)!=1 or len(source_names)!=1:heterogeneous.add(r.record_id)
        fam[r.record_id]=next(iter(fs)) if len(fs)==1 else None;groupmap[r.record_id]=tuple(sorted(groups))
        details.append({'record_id':r.record_id,'functional_groups':';'.join(sorted(groups)),'protected_resistance':r.record_id in protected,'heterogeneous_original_family_annotations':len(fs)!=1,'heterogeneous_original_function_groups':len(source_groups)!=1,'evidence':json.dumps(evidence),'represented_species':r.represented_species,'represented_assays':r.represented_assays,'scope_reason':'explicit functional annotation' if groups else 'outside predefined mechanism and named-essential scope; not evidence of biological irrelevance'})
    pd.DataFrame(details).to_csv(out/'target_function_evidence.csv',index=False)
    hits=pd.read_csv(compression/'alignments.tsv',sep='\t',names=['query','target','nident','qlen','tlen','qcov','tcov']);hits['identity']=hits.nident/hits[['qlen','tlen']].max(axis=1)
    if not hits[hits['query']==hits['target']].identity.eq(1).all():raise ValueError('Invalid identity alignments')
    links={(x.query,x.target):x.identity for x in hits.itertuples() if min(x.qcov,x.tcov)>=.9}
    order=frame.assign(_review=frame.reviewed.map(true)).sort_values(['_review','annotation_score','length','record_id'],ascending=[False,False,False,True]).record_id.tolist();table=frame.set_index('record_id');summary=[]
    core=set(PATTERNS)-{'functional_transport'}|{'resistance_annotation'}
    for tier,allowed in [('core',core),('core_transport',core|{'functional_transport'}),('core_transport_essential',core|{'functional_transport','functionally_named_essential'})]:
        eligible={k for k,g in groupmap.items() if set(g)&allowed}|protected
        for threshold in (.9,.5):
            reps=[];mapping=[];groups={}
            for pid in order:
                if pid not in eligible:continue
                signature=(str(table.loc[pid,'represented_species']),groupmap[pid],fam[pid],names[pid]);rep=pid;identity=1.
                supported=fam[pid] is not None and any(fam[pid][:3])
                if pid not in protected and pid not in heterogeneous and supported:
                    for other in groups.get(signature,[]):
                        score=max(links.get((pid,other),0),links.get((other,pid),0))
                        if score>=threshold:rep=other;identity=score;break
                if rep==pid:
                    reps.append(pid)
                    if supported and pid not in heterogeneous:groups.setdefault(signature,[]).append(pid)
                mapping.append({'member_id':pid,'representative_id':rep,'global_identity':identity})
            destination=out/f'{tier}_{int(threshold*100)}';destination.mkdir();selected=frame[frame.record_id.isin(reps)];selected.to_csv(destination/'proteins.csv',index=False);pd.DataFrame(mapping).to_csv(destination/'membership.csv',index=False)
            (destination/'targets.fasta').write_text(''.join(f'>{r.record_id}\n{r.sequence}\n' for r in selected.itertuples()))
            pd.DataFrame([{'record_id':pid,'in_scope':pid in eligible,'selected':pid in reps,'reason':'scope exclusion' if pid not in eligible else 'representative retained' if pid in reps else 'direct homolog representative'} for pid in order]).to_csv(destination/'decisions.csv',index=False)
            retained_functions=set().union(*(all_domains[x] for x in selected.record_id));baseline_functions=set().union(*(all_domains[x] for x in frame.record_id))
            item={'tier':tier,'identity':threshold,'in_scope_count':len(eligible),'selected_count':len(selected),'scope_excluded':len(frame)-len(eligible),'protected_retained':len(protected&set(reps)),'species_counts':selected.species.value_counts().to_dict(),'functional_group_counts':{g:sum(g in groupmap[x] for x in reps) for g in sorted(set().union(*map(set,groupmap.values())))},'annotation_ids_baseline':len(baseline_functions),'annotation_ids_retained':len(retained_functions),'lost_annotation_ids':sorted(baseline_functions-retained_functions)}
            (destination/'coverage.json').write_text(json.dumps(item,indent=2));summary.append({k:v for k,v in item.items() if k!='lost_annotation_ids'})
    if any(sha256_file(k)!=v for k,v in inputs.items()):raise ValueError('Inputs changed')
    (out/'manifest.json').write_text(json.dumps({'status':'constructed_pending_independent_review','baseline_count':len(frame),'rules':rules,'input_sha256':inputs,'summary':summary,'output_sha256':{str(f.relative_to(out)):sha256_file(f) for f in out.rglob('*') if f.is_file()}},indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
