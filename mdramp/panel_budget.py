"""Fixed-budget, score-independent evidence-coverage target panel."""
import argparse,json,os,tempfile,shutil,fcntl
from pathlib import Path
import pandas as pd
from src.utils.hash import sha256_file
SPECIES=('Ab','Kp','Pa','Sa')


def choose(frame,functions,protected,budgets):
    """Greedy within species; shared sequences count once under canonical species."""
    chosen=set(protected);decisions=[]
    for species in SPECIES:
        candidates=frame[frame.species==species].copy()
        current=chosen & set(candidates.record_id)
        if len(current)>budgets[species] or len(candidates)<budgets[species]:raise ValueError('Species budget cannot satisfy mandatory targets')
        covered=set().union(*(functions[p] for p in current)) if current else set()
        rows=candidates.set_index('record_id').to_dict('index')
        def quality(pid):
            r=rows[pid];return (-int(str(r['reviewed']).lower()=='true'),-float(r['annotation_score']),-int(r['length']),pid)
        for pid in sorted(current):decisions.append({'record_id':pid,'budget_species':species,'reason':'mandatory_any_original_member_resistance','new_function_ids':None})
        while len(current)<budgets[species]:
            remaining=set(rows)-current
            pid=min(remaining,key=lambda k:(-len(functions[k]-covered),*quality(k)))
            gain=len(functions[pid]-covered);covered.update(functions[pid]);current.add(pid);chosen.add(pid)
            decisions.append({'record_id':pid,'budget_species':species,'reason':'greedy_unique_function_annotation_coverage','new_function_ids':gain})
    return chosen,decisions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel-dir','compression-dir','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    if a.budget<4:raise ValueError('Budget must be at least four')
    panel=Path(a.panel_dir).resolve();compression=Path(a.compression_dir).resolve();out=Path(a.output_dir).resolve()
    if out.exists():raise ValueError('Output already exists')
    inputs={str(p):sha256_file(p) for p in [panel/'proteins.csv',panel/'panel_membership.csv',panel/'library_evidence.csv',compression/'manifest.json',compression/'functional_signatures.csv',compression/'original_member_evidence.csv',Path(__file__).resolve()]}
    archived=json.loads((compression/'manifest.json').read_text())
    for name in ('functional_signatures.csv','original_member_evidence.csv'):
        if archived['output_sha256'].get(name)!=sha256_file(compression/name):raise ValueError('Compression evidence source hash mismatch')
    rules=json.loads((compression/'rules.json').read_text())
    for name in ('proteins.csv','panel_membership.csv','library_evidence.csv'):
        if rules['input_sha256'].get(str(panel/name))!=sha256_file(panel/name):raise ValueError('Compression evidence does not match baseline panel')
    inputs[str(compression/'rules.json')]=sha256_file(compression/'rules.json')
    frame=pd.read_csv(panel/'proteins.csv').fillna({'gene_names':'','represented_assays':''})
    if frame.record_id.duplicated().any() or frame.sequence.duplicated().any() or set(frame.species)!=set(SPECIES):raise ValueError('Unique four-species baseline required')
    source=pd.read_csv(compression/'original_member_evidence.csv')
    if set(source.scoring_record_id)!=set(frame.record_id):raise ValueError('Original-source evidence incomplete')
    protected=set(source.loc[source.resistance_annotation.astype(str).str.lower()=='true','scoring_record_id'])
    signatures=pd.read_csv(compression/'functional_signatures.csv').fillna('');functions={r.record_id:set(r.annotation_ids.split(';'))-{''} for r in signatures.itertuples()}
    if set(frame.record_id)-functions.keys():raise ValueError('Missing function evidence')
    budgets={s:a.budget//4+int(i<a.budget%4) for i,s in enumerate(SPECIES)}
    selection_rule={'budget':a.budget,'species_budgets':budgets,'rule':'All any-original-member resistance targets mandatory; then greedy maximum uncovered Pfam/InterPro/EC ID count within canonical species. Ties: reviewed, annotation score, length descending; record ID ascending.','shared_sequence_accounting':'Each unique sequence consumes one slot in its canonical species; all represented species/assays remain recorded.','used_mic_or_peptide_scores':False,'purpose':'new_prospective_budgeted_coverage_panel','interpretation':'Annotation coverage is a proxy. Selection does not establish functional equivalence, complete biological coverage, or matching experimental isolates.'}
    chosen,decisions=choose(frame,functions,protected,budgets)
    if len(chosen)!=a.budget or not protected<=chosen:raise ValueError('Budget/protected invariant failed')
    selected=frame[frame.record_id.isin(chosen)].copy();selected['protected_any_source_resistance']=selected.record_id.isin(protected)
    coverage=[]
    for species in ('ALL',*SPECIES):
        group=frame if species=='ALL' else frame[frame.species==species];kept=selected if species=='ALL' else selected[selected.species==species]
        before=set().union(*(functions[x] for x in group.record_id));after=set().union(*(functions[x] for x in kept.record_id))
        coverage.append({'species':species,'baseline_proteins':len(group),'selected_proteins':len(kept),'baseline_function_ids':len(before),'retained_function_ids':len(after),'lost_function_ids':sorted(before-after),'annotation_coverage_fraction':len(after)/len(before) if before else None})
    categories=[]
    for column in ('represented_species','represented_assays','evidence_signature','gene_names'):
        before=set(frame[column].fillna('').astype(str));after=set(selected[column].fillna('').astype(str));categories.append({'dimension':column,'original_unique_values':len(before),'retained_unique_values':len(after),'lost_values':sorted(before-after)})
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/(out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():raise ValueError('Output exists')
        stage=Path(tempfile.mkdtemp(prefix='.'+out.name,dir=out.parent))
        try:
            (stage/'selection_rules.json').write_text(json.dumps(selection_rule,indent=2))
            selected.to_csv(stage/'proteins.csv',index=False)
            (stage/'targets.fasta').write_text(''.join(f'>{r.record_id}\n{r.sequence}\n' for r in selected.itertuples()))
            pd.DataFrame(decisions).to_csv(stage/'selection_decisions.csv',index=False)
            pd.DataFrame({'record_id':frame.record_id,'selected':frame.record_id.isin(chosen),'protected_any_source_resistance':frame.record_id.isin(protected)}).to_csv(stage/'baseline_selection.csv',index=False)
            source[source.scoring_record_id.isin(chosen)].to_csv(stage/'original_member_evidence.csv',index=False)
            membership=pd.read_csv(panel/'panel_membership.csv');membership[membership.scoring_record_id.isin(chosen)].to_csv(stage/'panel_membership.csv',index=False)
            (stage/'coverage.json').write_text(json.dumps({'function_coverage':coverage,'category_coverage':categories},indent=2))
            if any(sha256_file(k)!=v for k,v in inputs.items()):raise ValueError('Inputs changed')
            manifest={'schema_version':1,'status':'selected_pending_independent_stability_validation','historical_r0_panel':False,'actual_assay_isolate_match':False,'record_ids':selected.record_id.tolist(),'protein_count':len(selected),'baseline_count':len(frame),'protected_target_count':len(protected),'species_budgets':budgets,'inputs_sha256':inputs,'output_sha256':{f.name:sha256_file(f) for f in stage.iterdir()},'global_function_coverage':coverage[0]}
            (stage/'manifest.json').write_text(json.dumps(manifest,indent=2));os.rename(stage,out)
        finally:
            if stage.exists():shutil.rmtree(stage)
    print(json.dumps({'proteins':len(selected),'protected':len(protected),'function_coverage':coverage[0]['annotation_coverage_fraction'],'species_budgets':budgets}))
if __name__=='__main__':main()
