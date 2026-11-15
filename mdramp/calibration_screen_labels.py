"""User-assigned lowest-positive-dose MIC proxies, preserving original observations."""
import argparse,hashlib,json,math,os,re,shutil,tempfile
from pathlib import Path
import pandas as pd
from mdramp.calibration_data import parse_mic,intersect_constraints
from mdramp.calibration_best import minimum_interval,STRAINS
SEMANTICS='user_assigned_screen_mic'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def provenance_json(value):
    def convert(x):
        if isinstance(x,float) and math.isnan(x):return None
        if isinstance(x,float) and math.isinf(x):return 'inf' if x>0 else '-inf'
        if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [convert(v) for v in x]
        return x
    return json.dumps(convert(value),ensure_ascii=False,allow_nan=False)

def assign_observation(row):
    result=dict(row);evidence=json.loads(row['provenance']);quant=[];positive=[];negative=[];adjudications=[]
    for record in evidence:
        if 'user_adjudication' in record:
            adjudications.append(parse_mic(record['user_adjudication']['mic_ug_ml']));continue
        for cell in record.get('cells',[]):
            value=str(cell['value']).strip().replace('＋','+').replace('−','-').replace('－','-')
            if not value:continue
            if cell['header'].startswith('MIC_'):
                quant.append(parse_mic(value));continue
            dose=re.search(r'\b(32|256)\s*μg/mL',cell['header'])
            if not dose or value not in ('+','-'):raise ValueError('Unsupported screening provenance')
            (positive if value=='+' else negative).append(float(dose[1]))
    chosen=adjudications or quant
    if chosen:
        lo,hi,li,ui,status,reason=intersect_constraints(chosen)
        if status!='usable':raise ValueError('Conflicting quantitative MIC records require adjudication')
        source='user_adjudicated_quantitative_MIC' if adjudications else 'reported_quantitative_MIC'
    elif positive:
        dose=min(positive)
        if any(n>=dose for n in negative):raise ValueError('Nonmonotone screening results require adjudication')
        lo=hi=math.log2(dose);li=ui=True;source='user_assigned_lowest_positive_tested_dose'
    elif negative:
        lo=math.log2(max(negative));hi=math.inf;li=ui=False;source='right_censored_above_highest_negative_tested_dose'
    else:raise ValueError('Missing MIC and screening evidence')
    conflicts=[]
    if chosen:
        if any(lo>math.log2(d) or (lo==math.log2(d) and not li) for d in positive):conflicts.append('positive_screen_inconsistent_with_quantitative_MIC')
        if any(hi<=math.log2(d) for d in negative):conflicts.append('negative_screen_inconsistent_with_quantitative_MIC')
    for key in ['lower_log2','upper_log2','lower_inclusive','upper_inclusive','exact_log2','observation_type']:
        result['source_interval_'+key]=row[key]
    kind='exact' if lo==hi and li and ui else 'left_censored' if lo==-math.inf else 'right_censored' if hi==math.inf else 'interval'
    result.update(lower_log2=lo,upper_log2=hi,lower_inclusive=li,upper_inclusive=ui,status='usable',observation_type=kind,exact_log2=lo if kind=='exact' else None,exclusion_reason='',target_semantics=SEMANTICS,target_divisor=1,label_assignment_source=source,screen_quantitative_conflicts=';'.join(sorted(set(conflicts))))
    evidence.append({'user_assignment_policy':SEMANTICS,'rule':'explicit quantitative MIC/adjudication first; otherwise lowest positive tested concentration assigned exactly; all-negative remains right-censored','label_assignment_source':source,'screen_quantitative_conflicts':conflicts,'not_new_experimental_MIC_measurement':True})
    result['provenance']=provenance_json(evidence)
    return result


def prepare(source_dir,output_dir):
    source=Path(source_dir).resolve();out=Path(output_dir).resolve()
    if out.exists():raise ValueError('New output version required')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest.get('positive_symbol')!='plus' or manifest.get('target_semantics')==SEMANTICS:raise ValueError('Original plus-active source required')
    inputs={str(source/'manifest.json'):sha(source/'manifest.json'),str(Path(__file__).resolve()):sha(__file__)}
    for name,digest in manifest['output_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Original dataset hash mismatch')
        inputs[str(source/name)]=digest
    for name in ['calibration_data.py','calibration_best.py']:
        path=Path(__file__).resolve().parent/name;inputs[str(path)]=sha(path)
    sequences=pd.read_csv(source/'sequences.csv');original=pd.read_csv(source/'observations.csv')
    if 'target_semantics' in original or 'target_divisor' in original:raise ValueError('Already transformed source not accepted')
    if original.duplicated(['peptide_id','strain_id']).any() or sequences.peptide_id.duplicated().any():raise ValueError('Duplicate source identity')
    assigned=pd.DataFrame([assign_observation(r) for r in original.to_dict('records')]);best=[]
    for record in sequences.to_dict('records'):
        pid=record['peptide_id'];rows=assigned[assigned.peptide_id.eq(pid)];prior=original[original.peptide_id.eq(pid)]
        if len(rows)!=6 or set(rows.strain_id)!=set(STRAINS) or not rows.status.eq('usable').all():raise ValueError('Complete six-strain panel required')
        if not rows.cluster_id.eq(record['cluster_id']).all() or not rows.fold.eq(record['fold']).all():raise ValueError('Group assignment changed')
        lo,hi,li,ui=minimum_interval(rows.to_dict('records'));oldlo,oldhi,_,_=minimum_interval(prior.to_dict('records'))
        kind='exact' if lo==hi and li and ui else 'right_censored' if hi==math.inf else 'left_censored' if lo==-math.inf else 'interval'
        best.append(dict(peptide_id=pid,strain_id='best_panel',lower_log2=lo,upper_log2=hi,lower_inclusive=li,upper_inclusive=ui,status='usable',observation_type=kind,exact_log2=lo if kind=='exact' else None,cluster_id=record['cluster_id'],fold=record['fold'],source_strain_count=6,target_semantics=SEMANTICS,target_divisor=1,source_interval_lower_log2=oldlo,source_interval_upper_log2=oldhi,provenance=provenance_json(rows.to_dict('records')),exclusion_reason=''))
    best=pd.DataFrame(best);out.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.'+out.name,dir=out.parent))
    try:
        shutil.copyfile(source/'sequences.csv',stage/'sequences.csv');shutil.copyfile(source/'observations.csv',stage/'source_observations.csv')
        assigned.to_csv(stage/'strain_observations.csv',index=False);best.to_csv(stage/'observations.csv',index=False)
        if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('Source changed during assignment')
        summary={'status':'complete','task':'peptide_best_mic','target_semantics':SEMANTICS,'target_divisor':1,'policy':'Explicit user rule: lowest positive tested concentration is assigned as MIC proxy; reported quantitative MIC/adjudication takes priority; no-positive samples remain right-censored. Aggregate minimum over six strains before any later divisor.','not_new_experimental_measurements':True,'input_sha256':inputs,'unique_sequences':len(sequences),'source_observations':len(assigned),'strain_type_counts':assigned.observation_type.value_counts().to_dict(),'best_type_counts':best.observation_type.value_counts().to_dict(),'assignment_sources':assigned.label_assignment_source.value_counts().to_dict(),'quantitative_screen_conflicts':int(assigned.screen_quantitative_conflicts.ne('').sum()),'sequence_fold_file_preserved_exactly':sha(source/'sequences.csv')==sha(stage/'sequences.csv'),'output_sha256':{p.name:sha(p) for p in stage.iterdir()}}
        (stage/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');os.rename(stage,out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(json.dumps({k:summary[k] for k in ['unique_sequences','strain_type_counts','best_type_counts','assignment_sources','quantitative_screen_conflicts']},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();prepare(a.source_dir,a.output_dir)
if __name__=='__main__':main()
