"""Prespecified development selection and one-time confirmation of panel compression."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import spearmanr
from mdramp.calibration_train import fresh_output
from mdramp.protein_panel_build import sha,dump
from src.models.gated_integrator import integrate_numpy,GateParams


@torch.inference_mode()
def aggregate(logits, fixed):
    tensor=torch.as_tensor(np.array(logits,dtype=np.float32,copy=True))
    if tensor.ndim!=3 or tensor.shape[1]!=5 or tensor.shape[2]<5 or not torch.isfinite(tensor).all():raise ValueError('Invalid complete five-member pair scores')
    top=tensor.topk(5,dim=2).values;q=top.sigmoid()
    members=.8*(top.softmax(dim=2)*q).sum(dim=2)+.2*(1-(1-q).prod(dim=2))
    mu=members.mean(dim=1).numpy();sigma=members.std(dim=1,unbiased=True).numpy()
    cons=integrate_numpy(fixed.mu_prior.to_numpy(),fixed.mu_mem.to_numpy(),mu,GateParams(),fixed.sigma_prior.to_numpy(),fixed.sigma_mem.to_numpy(),sigma,1.)['p_cons']
    return dict(mu_int=mu,sigma_int=sigma,p_cons=cons)


def rho(x,y):
    if np.ptp(x)==0 or np.ptp(y)==0:return None
    value=float(spearmanr(x,y).statistic)
    return value if np.isfinite(value) else None


def measure(x,y,clusters,ids,protocol,seed,bootstrap=True):
    x=np.asarray(x);y=np.asarray(y)
    if x.shape!=y.shape or len(x)<protocol['top_k'] or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Missing or invalid cohort scores')
    k=protocol['top_k'];ids=np.asarray(ids)
    topx=np.lexsort((ids,-x))[:k];topy=np.lexsort((ids,-y))[:k]
    retained=len(set(ids[topx])&set(ids[topy]));corr=rho(x,y);delta=abs(y-x)
    values=[]
    if bootstrap:
        groups=[np.flatnonzero(np.asarray(clusters)==c) for c in sorted(set(clusters))]
        random=np.random.default_rng(seed)
        for _ in range(protocol['bootstrap_replicates']):
            indices=np.concatenate([groups[i] for i in random.integers(0,len(groups),len(groups))])
            v=rho(x[indices],y[indices])
            if v is not None:values.append(v)
    ci=list(map(float,np.quantile(values,[.025,.975]))) if values else [None,None]
    mean=float(delta.mean());p95=float(np.quantile(delta,.95))
    enough=len(values)>=.9*protocol['bootstrap_replicates'] if bootstrap else False
    passed=bool(corr is not None and corr>=protocol['minimum_spearman'] and enough and ci[0]>=protocol['minimum_cluster_bootstrap_lower_95'] and retained>=protocol['minimum_top_k_retained'] and mean<=protocol['maximum_mean_absolute_score_drift'] and p95<=protocol['maximum_p95_absolute_score_drift'])
    return dict(n=len(x),clusters=len(set(clusters)),spearman=corr,spearman_cluster_bootstrap_95=ci,valid_bootstrap_replicates=len(values),
                top_k=k,top_k_retained=retained,top_k_fraction=retained/k,mean_absolute_drift=mean,p95_absolute_drift=p95,max_absolute_drift=float(delta.max()),passes_all_gates=passed if bootstrap else None)


def validate_manifest(root,name='manifest.json'):
    root=Path(root);m=json.loads((root/name).read_text())
    if m.get('status')!='complete':raise ValueError('Complete artifact required: '+str(root))
    for key,value in m['output_sha256'].items():
        if sha(root/key)!=value:raise ValueError('Artifact changed: '+str(root/key))
    return m


def random_representatives(mapping, baseline, selected, seed, width):
    """Same groups/budget; random representative, not an unstructured random-panel null."""
    table=baseline.set_index('record_id');random=np.random.default_rng(seed);chosen=[];eligible_alternatives=0
    for representative,group in mapping.groupby('representative_id',sort=True):
        r=table.loc[representative]
        protected=group[group.protected_resistance.astype(str).str.lower().eq('true')].member_id.tolist()
        if protected:
            if protected!=[representative]:raise ValueError('Protected member removed by compression')
            options=[representative]
        else:
            # Compression already enforces functional and evidence concordance.
            options=[]
            for pid in group.member_id:
                q=table.loc[pid]
                if all(str(q[c])==str(r[c]) for c in ['represented_species','represented_assays','evidence_signature']) and int(q.length)//width==int(r.length)//width:options.append(pid)
            options=sorted(options)
            if not options:raise ValueError('Representative not eligible for its own matched control')
        eligible_alternatives+=int(len(options)>1)
        chosen.append(options[int(random.integers(0,len(options)))])
    if len(chosen)!=len(selected) or len(chosen)!=len(set(chosen)):raise ValueError('Matched control size mismatch')
    return chosen,eligible_alternatives


def evaluate(a):
    torch.set_num_threads(a.threads)
    cache=Path(a.cache_dir).absolute();compression=Path(a.compression_dir).absolute();probes=Path(a.probes_dir).absolute()
    cache_manifest=validate_manifest(cache);compression_manifest=validate_manifest(compression)
    protocol=json.loads(Path(a.protocol).read_text());probe_manifest=json.loads((probes/'manifest.json').read_text())
    if probe_manifest.get('status')!='frozen_label_free_cohorts' or probe_manifest['protocol']!=protocol:raise ValueError('Protocol differs from frozen probe cohort design')
    for name,digest in probe_manifest['output_sha256'].items():
        if sha(probes/name)!=digest:raise ValueError('Frozen probes changed')
    inputs={str(Path(p).absolute()):sha(p) for p in [a.protocol,cache/'manifest.json',compression/'manifest.json',probes/'manifest.json',__file__]}
    base=Path(__file__).resolve().parents[1]
    for path in [base/'src/models/gated_integrator.py',base/'mdramp/calibration_train.py']:inputs[str(path)]=sha(path)
    inputs.update({str(cache/k):v for k,v in cache_manifest['output_sha256'].items()})
    inputs.update({str(compression/k):v for k,v in compression_manifest['output_sha256'].items()})
    inputs.update({str(probes/k):v for k,v in probe_manifest['output_sha256'].items()})
    peptides=pd.read_csv(cache/'peptides.csv');targets=pd.read_csv(cache/'targets.csv').fillna('');fixed=pd.read_csv(cache/'fixed_experts.csv')
    cohorts=pd.read_csv(probes/'cohorts.csv');probe_table=pd.read_csv(probes/'probes.csv')
    if peptides.record_id.tolist()!=fixed.record_id.tolist() or set(peptides.record_id)!=set(probe_table.record_id):raise ValueError('Probe cache coverage/order mismatch')
    if peptides.record_id.duplicated().any() or targets.record_id.duplicated().any():raise ValueError('Duplicate cache IDs')
    expected=probe_table.set_index('record_id').sequence
    if peptides.sequence.tolist()!=expected.loc[peptides.record_id].tolist():raise ValueError('Probe sequence mismatch')
    if cohorts[['cohort','record_id']].duplicated().any() or set(cohorts.record_id)!=set(probe_table.record_id):raise ValueError('Invalid cohort membership')
    if cohorts.sequence.tolist()!=expected.loc[cohorts.record_id].tolist():raise ValueError('Cohort sequence mismatch')
    expected_clusters=probe_table.set_index('record_id').cluster_id
    if cohorts.cluster_id.tolist()!=expected_clusters.loc[cohorts.record_id].tolist():raise ValueError('Cohort cluster mismatch')
    baseline_fasta=[(r.id,str(r.seq)) for r in SeqIO.parse(compression/'baseline.fasta','fasta')]
    if list(zip(targets.record_id,targets.sequence))!=baseline_fasta:raise ValueError('Compression baseline differs from scored protein identities or sequences')
    logits=np.load(cache/'logits.npy',mmap_mode='r')
    if list(logits.shape)!=[len(peptides),5,len(targets)]:raise ValueError('Complete logit array required')
    row_index={p:i for i,p in enumerate(peptides.record_id)};col_index={p:i for i,p in enumerate(targets.record_id)}
    panels={}
    for summary in compression_manifest['summary']:
        key='identity_'+str(round(100*summary['identity']));selected=pd.read_csv(compression/key/'proteins.csv').fillna('');mapping=pd.read_csv(compression/key/'membership.csv')
        if selected.record_id.duplicated().any() or not set(selected.record_id)<=set(targets.record_id) or len(selected)!=summary['proteins']:raise ValueError('Invalid compressed subset')
        if selected.sequence.tolist()!=targets.set_index('record_id').sequence.loc[selected.record_id].tolist():raise ValueError('Compressed subset sequence mismatch')
        if mapping.member_id.duplicated().any() or set(mapping.member_id)!=set(targets.record_id) or set(mapping.representative_id)!=set(selected.record_id):raise ValueError('Invalid compressed membership')
        panels[key]=(selected,mapping)
    with fresh_output(a.output_dir) as out:
        results={};baseline={};score_tables=[]
        def compare(name,ids,cohort,bootstrap=True):
            group=cohorts[cohorts.cohort.eq(cohort)].sort_values('record_id');rows=[row_index[x] for x in group.record_id]
            if cohort not in baseline:baseline[cohort]=aggregate(logits[rows],fixed.iloc[rows])
            cols=[col_index[x] for x in ids];scores=aggregate(logits[rows][:,:,cols],fixed.iloc[rows]);base=baseline[cohort]
            seed=protocol['seed']+int(hashlib.sha256(cohort.encode()).hexdigest()[:6],16)
            metrics={m:measure(base[m],scores[m],group.cluster_id.to_numpy(),group.record_id.to_numpy(),protocol,seed,bootstrap) for m in ['mu_int','p_cons']}
            metrics['mean_absolute_sigma_int_drift']=float(abs(base['sigma_int']-scores['sigma_int']).mean())
            metrics['passes_all_gates']=all(metrics[m]['passes_all_gates'] for m in ['mu_int','p_cons']) if bootstrap else None
            frame=group[['cohort','record_id','sequence','cluster_id']].copy();frame['panel']=name
            for metric in ['mu_int','sigma_int','p_cons']:frame[metric]=scores[metric];frame['baseline_'+metric]=base[metric]
            return metrics,frame
        for name,(selected,mapping) in panels.items():
            results[name]={'protein_count':len(selected),'development':{}}
            for cohort in protocol['development_cohorts']:
                metrics,scores=compare(name,selected.record_id.tolist(),cohort);results[name]['development'][cohort]=metrics;score_tables.append(scores)
            results[name]['passes_development']=all(v['passes_all_gates'] for v in results[name]['development'].values())
            print(f'{name} development gates: {results[name]["passes_development"]}',flush=True)
        pd.concat(score_tables,ignore_index=True).to_csv(out/'development_scores.csv',index=False);dump(out/'development_metrics.json',results)
        passed=[name for name in panels if results[name]['passes_development']]
        choice=min(passed,key=lambda name:(len(panels[name][0]),-int(name.split('_')[1]))) if passed else None
        selection={'stage':'development_choice_frozen_before_confirmation','chosen_panel':choice,'chosen_count':len(panels[choice][0]) if choice else len(targets),
                   'reason':'smallest_panel_passing_both_development_cohorts' if choice else 'no_compressed_panel_passed_development',
                   'protocol_sha256':sha(a.protocol),'development_metrics_sha256':sha(out/'development_metrics.json')}
        dump(out/'development_choice.json',selection);selection_hash=sha(out/'development_choice.json')
        events=[dict(event='development_choice_frozen',sha256=selection_hash)]
        confirmation=None;adopted=targets;status='no_compression_passed_development'
        if choice:
            confirmation,scores=compare(choice,panels[choice][0].record_id.tolist(),protocol['confirmation_cohort'])
            scores.to_csv(out/'confirmation_scores.csv',index=False);dump(out/'confirmation_metrics.json',confirmation)
            events.append(dict(event='one_time_confirmation',chosen_panel=choice,passed=confirmation['passes_all_gates']))
            if confirmation['passes_all_gates']:adopted=panels[choice][0];status='compression_confirmed_for_sequence_cohort_stability'
            else:status='compression_not_confirmed_baseline_retained'
            audit,audit_scores=compare(choice,panels[choice][0].record_id.tolist(),protocol['audit_only_cohort'])
            dump(out/'private_sequence_audit_metrics.json',audit);audit_scores.to_csv(out/'private_sequence_audit_scores.csv',index=False)
        controls=[];control_members=[]
        for name,(selected,mapping) in panels.items():
            for seed in protocol['random_control_seeds']:
                ids,alternatives=random_representatives(mapping,targets,selected,seed,protocol['length_bin_width'])
                control_members.extend(dict(panel=name,seed=seed,record_id=pid) for pid in ids)
                for cohort in protocol['development_cohorts']:
                    metrics,_=compare(name+'_random_'+str(seed),ids,cohort,bootstrap=False)
                    controls.append(dict(panel=name,seed=seed,cohort=cohort,alternative_groups=alternatives,metrics=metrics))
        dump(out/'matched_representative_controls.json',{'interpretation':'Controls test representative choice within the same compressed family groups and budget. They do not independently identify the full-panel versus subset opportunity-number effect. They do not select the main panel.','results':controls})
        pd.DataFrame(control_members).to_csv(out/'matched_control_members.csv',index=False)
        adopted.to_csv(out/'recommended_proteins.csv',index=False)
        if sha(out/'development_choice.json')!=selection_hash:raise ValueError('Development choice changed after confirmation')
        dump(out/'events.json',events)
        if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('Evaluation inputs changed')
        dump(out/'manifest.json',dict(status=status,recommended_count=len(adopted),development_choice=selection,confirmation=confirmation,
            no_mic_or_Rt_evaluated=True,private_sequences_used_for_selection=False,controls_not_used_for_selection=True,
            input_sha256=inputs,limitations=protocol['scope_limitations']+['Matched representative controls do not fully separate opportunity-number bias.','Stable expert scores do not establish clinical or biological equivalence.'],
            output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({'status':status,'recommended_count':len(adopted)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['cache-dir','compression-dir','probes-dir','protocol','output-dir']:p.add_argument('--'+n,required=True)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    if a.threads<1:raise ValueError('Positive thread count required')
    evaluate(a)


if __name__=='__main__':main()
