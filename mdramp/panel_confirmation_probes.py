"""Freeze a fresh public-test cohort without MIC or score access."""
import argparse,json,hashlib,tempfile,os
from pathlib import Path
import numpy as np
import pandas as pd
from mdramp.panel_probes import make_identity,sequence_id
from mdramp.protein_panel_build import sha

def prepare(public_path,old_path,protocol_path,output_dir):
    out=Path(output_dir).absolute()
    if out.exists():raise FileExistsError(out)
    paths=[Path(p).absolute() for p in [public_path,old_path,protocol_path,__file__]]
    hashes={str(p):sha(p) for p in paths}
    protocol=json.loads(Path(protocol_path).read_text())
    if protocol['seed']!=20261114 or protocol['public_cohort_size']!=200 or protocol['candidate_budget']!=500:raise ValueError('Unexpected frozen protocol')
    public=pd.read_csv(public_path,usecols=['sequence','cluster_id','split'],dtype=str)
    old=pd.read_csv(old_path,usecols=['sequence'],dtype=str)
    if public.sequence.duplicated().any() or old.sequence.duplicated().any() or len(old)!=827:raise ValueError('Require unique public and exact prior 827 sequences')
    if (public.groupby('cluster_id').split.nunique()>1).any():raise ValueError('Cluster crosses historical splits')
    identity=make_identity();test=public[public.split.eq('test')].copy()
    if not test.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10,50}').all():raise ValueError('Invalid public test sequence')
    blocked=set(public.loc[public.sequence.isin(old.sequence),'cluster_id']);audit=[]
    # Test every member, not just representative: an old homolog excludes its entire source cluster.
    for row in test.itertuples():
        for previous in old.sequence:
            if identity(row.sequence,previous)>=.9:
                blocked.add(row.cluster_id);audit.append({'sequence':row.sequence,'cluster_id':row.cluster_id,'old_sequence':previous,'reason':'old_827_at_least_90pct_identity'});break
    candidates=test[~test.cluster_id.isin(blocked)].copy()
    candidates['record_id']=candidates.sequence.map(sequence_id)
    candidates=candidates.sort_values('record_id').drop_duplicates('cluster_id').sort_values('cluster_id')
    order=np.random.default_rng(protocol['seed']).permutation(len(candidates));chosen=[]
    for index in order:
        row=candidates.iloc[index]
        if any(identity(row.sequence,previous['sequence'])>=.9 for previous in chosen):continue
        chosen.append(row.to_dict())
        if len(chosen)==200:break
    if len(chosen)!=200:raise ValueError('Insufficient fresh isolated public test clusters; thresholds unchanged')
    result=pd.DataFrame(chosen)[['record_id','sequence','cluster_id','split']]
    if any(identity(s,t)>=.9 for s in result.sequence for t in old.sequence):raise AssertionError('Old cohort homology overlap')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix='.confirmation-') as temp:
        stage=Path(temp)/'published';stage.mkdir()
        result.to_csv(stage/'probes.csv',index=False)
        members=result.copy();members.insert(0,'cohort','public_test_confirmation_v2');members.to_csv(stage/'cohorts.csv',index=False)
        pd.DataFrame(audit).to_csv(stage/'sampling_exclusions.csv',index=False)
        pd.DataFrame({'cluster_id':sorted(blocked)}).to_csv(stage/'excluded_public_clusters.csv',index=False)
        if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('Input changed during cohort freeze')
        manifest={'status':'frozen_label_free_cohorts','input_sha256':hashes,'protocol':protocol,'old_probes_path':str(Path(old_path).absolute()),'old_probes_sha256':sha(old_path),'public_path':str(Path(public_path).absolute()),'protocol_path':str(Path(protocol_path).absolute()),'protocol_sha256':sha(protocol_path),'counts':{'public_test_confirmation_v2':200},'unique_probe_count':200,'cluster_count':200,'no_mic_values_loaded':True,'no_model_scores_loaded':True,'independence_audit':{'old_unique_sequence_count':827,'new_unique_sequence_count':200,'old_overlap_at_identity_ge_0_90':0,'new_pair_overlap_at_identity_ge_0_90':0,'public_test_rows':len(test),'excluded_public_clusters':len(blocked),'eligible_clusters':len(candidates),'method':'same global aligned identity as prior protocol; every public test cluster member screened against all old827; entire matching clusters excluded; selected representatives pairwise screened'},'output_sha256':{p.name:sha(p) for p in stage.iterdir()}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2));os.rename(stage,out)
    print(json.dumps(manifest['independence_audit']),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['public','old-probes','protocol','output-dir']:p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.public,a.old_probes,a.protocol,a.output_dir)

if __name__=='__main__':main()
