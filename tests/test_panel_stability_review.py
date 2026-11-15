"""Independent counterexamples for frozen development-only panel selection."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from mdramp import panel_stability as ps


def protocol():
    return dict(seed=31,top_k=2,bootstrap_replicates=100,minimum_spearman=.95,minimum_cluster_bootstrap_lower_95=.9,
        minimum_top_k_retained=2,maximum_mean_absolute_score_drift=.02,maximum_p95_absolute_score_drift=.05,
        development_cohorts=['devA','devB'],confirmation_cohort='test',audit_only_cohort='audit',random_control_seeds=[11,12],length_bin_width=100,scope_limitations=[])


def test_constant_correlation_cannot_pass_and_controls_have_no_gate():
    p=protocol();ids=np.array(['a','b','c','d']);x=np.ones(4)
    result=ps.measure(x,x,ids,ids,p,1)
    assert result['spearman'] is None and not result['passes_all_gates']
    y=np.arange(4,dtype=float)
    result=ps.measure(y,y,ids,ids,p,1,bootstrap=False)
    assert result['passes_all_gates'] is None and result['valid_bootstrap_replicates']==0


def test_random_controls_keep_budget_and_protected_representative():
    base=pd.DataFrame([dict(record_id=x,represented_species='Ab',represented_assays='WT',evidence_signature='100',length=40) for x in 'abcd'])
    mapping=pd.DataFrame({'member_id':list('abcd'),'representative_id':['a','a','c','c'],'protected_resistance':[True,False,False,False]})
    for seed in range(10):
        ids,n=ps.random_representatives(mapping,base,base[base.record_id.isin(['a','c'])],seed,100)
        assert len(set(ids))==2 and 'a' in ids and 'b' not in ids and n==1
    mapping.loc[1,'protected_resistance']=True
    with pytest.raises(ValueError,match='Protected'):ps.random_representatives(mapping,base,base.iloc[:2],1,100)


def test_failed_confirmation_never_selects_second_best_panel(tmp_path):
    cache=tmp_path/'cache';compression=tmp_path/'compression';probes=tmp_path/'probes'
    for p in [cache,compression,probes]:p.mkdir()
    prot=protocol();cfg=tmp_path/'protocol.json';cfg.write_text(json.dumps(prot))
    targets=pd.DataFrame([dict(record_id=f't{i}',sequence='ACDEFGHIK'+chr(65+i),represented_species='Ab',represented_assays='WT',evidence_signature=100,length=10) for i in range(7)])
    targets.to_csv(cache/'targets.csv',index=False)
    rows=[dict(record_id=f'p{i:02}',sequence='ACDEFGHIK'+str(i),cluster_id=f'c{i}',cohort=c) for i,c in enumerate(sum(([c]*4 for c in ['devA','devB','test','audit']),[]))]
    peptides=pd.DataFrame(rows);peptides[['record_id','sequence']].to_csv(cache/'peptides.csv',index=False)
    peptides.to_csv(probes/'cohorts.csv',index=False);peptides[['record_id','sequence','cluster_id']].to_csv(probes/'probes.csv',index=False)
    fixed=pd.DataFrame(dict(record_id=peptides.record_id,mu_prior=.6,sigma_prior=.01,mu_mem=.6,sigma_mem=.01));fixed.to_csv(cache/'fixed_experts.csv',index=False)
    logits=np.zeros((16,5,7),np.float32)
    for i in range(16):logits[i,:,:]=-2+(i%4)*.3
    # Only the frozen confirmation cohort exposes the smaller panel's failure.
    for i in range(8,12):logits[i,:,6]=4-(i%4)*.3
    np.save(cache/'logits.npy',logits)
    (compression/'baseline.fasta').write_text(''.join(f'>{r.record_id}\n{r.sequence}\n' for r in targets.itertuples()))
    summaries=[]
    for name,ids,identity in [('identity_90',[0,1,2,3,4,6],.9),('identity_50',[0,1,2,3,4],.5)]:
        d=compression/name;d.mkdir();selected=targets.iloc[ids];selected.to_csv(d/'proteins.csv',index=False)
        mapping=pd.DataFrame(dict(member_id=targets.record_id,representative_id=[f't{i}' if i in ids else 't0' for i in range(7)],protected_resistance=False));mapping.to_csv(d/'membership.csv',index=False)
        summaries.append(dict(identity=identity,proteins=len(ids)))
    for root,extra in [(cache,{}),(compression,{'summary':summaries}),(probes,{'protocol':prot})]:
        m=dict(status='frozen_label_free_cohorts' if root==probes else 'complete',output_sha256={str(p.relative_to(root)):ps.sha(p) for p in root.rglob('*') if p.is_file()},**extra)
        (root/'manifest.json').write_text(json.dumps(m))
    out=tmp_path/'out';ps.evaluate(SimpleNamespace(cache_dir=cache,compression_dir=compression,probes_dir=probes,protocol=cfg,output_dir=out,threads=1))
    result=json.loads((out/'manifest.json').read_text())
    assert result['development_choice']['chosen_panel']=='identity_50'
    assert result['status']=='compression_not_confirmed_baseline_retained' and result['recommended_count']==7
    assert set(pd.read_csv(out/'confirmation_scores.csv').panel)=={'identity_50'}
    events=json.loads((out/'events.json').read_text());assert [e['event'] for e in events]==['development_choice_frozen','one_time_confirmation']
    controls=json.loads((out/'matched_representative_controls.json').read_text())['results']
    assert len(controls)==2*2*2 and {x['cohort'] for x in controls}=={'devA','devB'}
    assert all(x['metrics']['passes_all_gates'] is None for x in controls)
    assert result['no_mic_or_Rt_evaluated'] and not result['private_sequences_used_for_selection']


def test_aggregate_is_per_member_top5_then_unbiased_ensemble_std():
    logits=np.array([[[.1*j+.2*m for j in range(7)] for m in range(5)]],np.float32)
    fixed=pd.DataFrame(dict(mu_prior=[.6],sigma_prior=[.02],mu_mem=[.7],sigma_mem=[.03]))
    result=ps.aggregate(logits,fixed)
    members=[]
    for row in logits[0]:
        top=np.sort(row)[-5:].astype(np.float64);q=1/(1+np.exp(-top));w=np.exp(top-top.max());w/=w.sum()
        members.append(.8*(w*q).sum()+.2*(1-np.prod(1-q)))
    assert result['mu_int'][0]==pytest.approx(np.mean(members),abs=2e-7)
    assert result['sigma_int'][0]==pytest.approx(np.std(members,ddof=1),abs=2e-7)
    assert result['sigma_int'][0]!=pytest.approx(np.std(members,ddof=0),abs=1e-5)
