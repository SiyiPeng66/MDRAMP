"""Evaluate one frozen budget panel without test-driven reselection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mdramp.panel_stability import aggregate,measure,validate_manifest
from mdramp.panel_probes import make_identity
from mdramp.protein_panel_build import sha,dump
from mdramp.calibration_train import fresh_output


def bind_outputs(root,manifest,binding):
    for name,digest in manifest['output_sha256'].items():
        path=(root/name).absolute()
        if sha(path)!=digest:raise ValueError('Artifact output changed')
        binding[str(path)]=digest


def validate_confirmation(root,protocol_path,peptides,cache_manifest,binding):
    """Verify provenance and actual sequences, never trusting claimed cluster IDs."""
    root=Path(root).absolute();protocol=json.loads(Path(protocol_path).read_text())
    mp=root/'manifest.json';manifest=json.loads(mp.read_text());binding[str(mp)]=sha(mp)
    if manifest.get('status')!='frozen_label_free_cohorts' or manifest.get('protocol')!=protocol or manifest.get('protocol_sha256')!=sha(protocol_path):raise ValueError('Confirmation protocol mismatch')
    if protocol.get('candidate_budget')!=500 or protocol.get('public_cohort_size')!=200 or protocol.get('prior_confirmation_used_as_independent') is not False:raise ValueError('Fresh 500-panel confirmation protocol required')
    bind_outputs(root,manifest,binding)
    for name,digest in manifest['input_sha256'].items():
        path=Path(name).absolute()
        if sha(path)!=digest:raise ValueError('Confirmation input changed')
        binding[str(path)]=digest
    oldpath=Path(manifest['old_probes_path']).absolute();publicpath=Path(manifest['public_path']).absolute()
    for path in [oldpath,publicpath,Path(protocol_path).absolute()]:
        if manifest['input_sha256'].get(str(path))!=sha(path):raise ValueError('Unbound confirmation source')
    if manifest.get('old_probes_sha256')!=sha(oldpath):raise ValueError('Old probes changed')
    frozen=pd.read_csv(root/'probes.csv',dtype=str);old=pd.read_csv(oldpath,usecols=['sequence'],dtype=str)
    public=pd.read_csv(publicpath,usecols=['sequence','cluster_id','split'],dtype=str)
    if len(frozen)!=200 or frozen.record_id.duplicated().any() or frozen.sequence.duplicated().any() or frozen.cluster_id.duplicated().any() or not frozen.split.eq('test').all():raise ValueError('Exactly 200 independent test probes required')
    if len(old)!=827 or old.sequence.duplicated().any() or public.sequence.duplicated().any() or (public.groupby('cluster_id').split.nunique()>1).any():raise ValueError('Invalid public/old source isolation')
    if not frozen.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10,50}').all() or not old.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10,50}').all():raise ValueError('Invalid confirmation sequences')
    if len(peptides)!=200 or peptides.record_id.duplicated().any() or set(peptides.record_id)!=set(frozen.record_id):raise ValueError('Cache probe coverage mismatch')
    expected=frozen.set_index('record_id')
    for column in ['sequence','cluster_id','split']:
        if column not in peptides or peptides[column].astype(str).tolist()!=expected.loc[peptides.record_id,column].tolist():raise ValueError('Cache probe identity mismatch')
    if cache_manifest.get('binding',{}).get('input_sha256',{}).get(str(root/'probes.csv'))!=sha(root/'probes.csv'):raise ValueError('Cache not bound to frozen confirmation probes')
    source=public.set_index('sequence')
    if not set(frozen.sequence)<=set(source.index):raise ValueError('Confirmation absent from public source')
    if source.loc[frozen.sequence,'cluster_id'].tolist()!=frozen.cluster_id.tolist() or not source.loc[frozen.sequence,'split'].eq('test').all():raise ValueError('Confirmation not from public test split')
    identity=make_identity();sequences=frozen.sequence.tolist()
    # Check all original members of selected public clusters against prior probes.
    for sequence in public.loc[public.cluster_id.isin(frozen.cluster_id),'sequence']:
        if any(identity(sequence,previous)>=.9 for previous in old.sequence):raise ValueError('Confirmation overlaps prior probe homology')
    for i,sequence in enumerate(sequences):
        if any(identity(sequence,previous)>=.9 for previous in sequences[:i]):raise ValueError('Confirmation internal homology overlap')
    return {'old_sequences':len(old),'confirmation_sequences':len(frozen),'actual_sequence_identity_threshold':.9,'old_overlap':0,'internal_overlap':0}


def evaluate(a):
    torch.set_num_threads(2)
    panel=Path(a.panel).absolute();cache=Path(a.cache_dir).absolute();protocol=json.loads(Path(a.protocol).read_text())
    manifest=validate_manifest(cache);pm_path=Path(a.panel_manifest or panel.parent/'manifest.json').absolute();pm=json.loads(pm_path.read_text())
    binding={str(panel):sha(panel),str(Path(a.protocol).absolute()):sha(a.protocol),str(Path(__file__).absolute()):sha(__file__),str(cache/'manifest.json'):sha(cache/'manifest.json'),str(pm_path):sha(pm_path)}
    for name in ['mdramp/panel_stability.py','mdramp/panel_probes.py','mdramp/calibration_train.py','src/models/gated_integrator.py']:
        f=Path(__file__).resolve().parents[1]/name;binding[str(f)]=sha(f)
    bind_outputs(cache,manifest,binding);bind_outputs(pm_path.parent,pm,binding)
    if pm.get('status')!='selected_pending_independent_stability_validation' or pm.get('protein_count')!=500 or pm['output_sha256'].get(panel.name)!=sha(panel):raise ValueError('Frozen budget panel manifest required')
    if protocol.get('fixed_candidate_path')!=str(panel) or protocol.get('fixed_candidate_sha256')!=sha(panel) or protocol.get('fixed_candidate_manifest_sha256')!=sha(pm_path):raise ValueError('Candidate differs from preregistered confirmation panel')
    for path,digest in pm['inputs_sha256'].items():
        if sha(path)!=digest:raise ValueError('Panel source changed')
        binding[str(Path(path).absolute())]=digest
    selected=pd.read_csv(panel);targets=pd.read_csv(cache/'targets.csv');peptides=pd.read_csv(cache/'peptides.csv',dtype=str);fixed=pd.read_csv(cache/'fixed_experts.csv')
    if len(selected)!=500 or selected.record_id.duplicated().any() or selected.sequence.duplicated().any() or targets.record_id.duplicated().any():raise ValueError('Exactly 500 unique targets required')
    lookup=targets.set_index('record_id');ids=selected.record_id.tolist()
    if not set(ids)<=set(lookup.index) or selected.sequence.tolist()!=lookup.loc[ids].sequence.tolist():raise ValueError('Panel sequence identity mismatch')
    evidence_paths=[Path(p) for p in pm['inputs_sha256'] if Path(p).name=='original_member_evidence.csv']
    if len(evidence_paths)!=1:raise ValueError('Original resistance evidence required')
    evidence=pd.read_csv(evidence_paths[0]);protected=set(evidence.loc[evidence.resistance_annotation.astype(str).str.lower().eq('true'),'scoring_record_id'])
    if len(protected)!=pm.get('protected_target_count') or not protected<=set(ids):raise ValueError('Protected resistance targets lost')
    baseline_paths=[Path(p) for p in pm['inputs_sha256'] if Path(p).name=='proteins.csv']
    if len(baseline_paths)!=1:raise ValueError('Unambiguous original panel required')
    baseline=pd.read_csv(baseline_paths[0])
    if list(zip(targets.record_id,targets.sequence))!=list(zip(baseline.record_id,baseline.sequence)):raise ValueError('Scored baseline differs from selection baseline')
    if peptides.record_id.tolist()!=fixed.record_id.astype(str).tolist():raise ValueError('Probe/fixed expert alignment missing')
    isolation=validate_confirmation(a.cohort_dir,a.protocol,peptides,manifest,binding)
    with fresh_output(a.output_dir) as out:
        dump(out/'frozen_panel.json',{'panel_count':500,'panel_sha256':sha(panel),'selection':'score-independent evidence coverage; no reselection after confirmation','input_sha256':binding})
        freeze=sha(out/'frozen_panel.json');logits=np.load(cache/'logits.npy',mmap_mode='r')
        if logits.shape!=(len(peptides),5,len(targets)):raise ValueError('Logit shape mismatch')
        col_index={v:i for i,v in enumerate(targets.record_id)};cols=[col_index[x] for x in ids]
        baseline=aggregate(logits,fixed);candidate=aggregate(logits[:,:,cols],fixed)
        ties=[hashlib.sha256(s.encode()).hexdigest() for s in peptides.sequence]
        metrics={k:measure(baseline[k],candidate[k],peptides.cluster_id.to_numpy(),ties,protocol,protocol['seed']) for k in ['mu_int','p_cons']}
        passed=all(x['passes_all_gates'] for x in metrics.values());table=peptides.copy()
        for k in candidate:table['baseline_'+k]=baseline[k];table['panel500_'+k]=candidate[k]
        table.to_csv(out/'scores.csv',index=False);dump(out/'metrics.json',metrics)
        if any(sha(k)!=v for k,v in binding.items()) or sha(out/'frozen_panel.json')!=freeze:raise ValueError('Inputs changed')
        dump(out/'manifest.json',{'status':'complete','passes_confirmation':passed,'panel_count':500,'frozen_panel_sha256':freeze,'input_sha256':binding,'independence_verified':isolation,'no_mic_or_Rt_evaluated':True,'interpretation':'Sequence-cohort expert-score stability only; not MIC or antibacterial efficacy. Failure does not trigger test-driven reselection.','output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps({'passes_confirmation':passed,'count':500}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['panel','cache-dir','protocol','cohort-dir','output-dir']:p.add_argument('--'+k,required=True)
    p.add_argument('--panel-manifest');evaluate(p.parse_args())
if __name__=='__main__':main()
