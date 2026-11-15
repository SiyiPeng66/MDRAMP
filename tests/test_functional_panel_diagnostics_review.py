"""Keep exploratory diagnostics restricted to frozen development rows."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from mdramp import functional_panel_diagnostics as diag


def fixture(tmp_path):
    cache=tmp_path/'cache';cohort=tmp_path/'cohort';panels=tmp_path/'panels';sub=panels/'core_90'
    for path in [cache,cohort,sub]:path.mkdir(parents=True)
    protocol={'development_cohorts':['generated_development','public_validation_development'],'seed':1,'top_k':1,'bootstrap_replicates':20,'minimum_spearman':.95,'minimum_cluster_bootstrap_lower_95':.9,'minimum_top_k_retained':1,'maximum_mean_absolute_score_drift':.02,'maximum_p95_absolute_score_drift':.05}
    pp=tmp_path/'protocol.json';pp.write_text(json.dumps(protocol))
    peptides=pd.DataFrame({'record_id':[f'p{i}' for i in range(6)],'sequence':[f'ACDEFGHIK{i}' for i in range(6)],'cluster_id':[f'c{i}' for i in range(6)]})
    groups=peptides.copy();groups['cohort']=['generated_development']*2+['public_validation_development']*2+['public_test_confirmation']*2
    groups.to_csv(cohort/'cohorts.csv',index=False);peptides.to_csv(cache/'peptides.csv',index=False)
    targets=pd.DataFrame({'record_id':[f't{i}' for i in range(6)],'sequence':[f'ACDEFGHIK{i}' for i in range(6)]});targets.to_csv(cache/'targets.csv',index=False);targets.iloc[:5].to_csv(sub/'proteins.csv',index=False)
    pd.DataFrame({'record_id':peptides.record_id,'mu_prior':.6,'mu_mem':.6,'sigma_prior':.01,'sigma_mem':.01}).to_csv(cache/'fixed_experts.csv',index=False)
    logits=np.zeros((6,5,6),np.float32)
    logits[1]=1;logits[3]=1;logits[4:]=np.nan # Reading confirmation in aggregate must fail.
    np.save(cache/'logits.npy',logits)
    for root,status,extra in [(cache,'complete',{}),(cohort,'frozen_label_free_cohorts',{'protocol':protocol}),(panels,'constructed_pending_independent_review',{'input_sha256':{}})]:
        (root/'manifest.json').write_text(json.dumps({'status':status,'output_sha256':{str(p.relative_to(root)):diag.sha(p) for p in root.rglob('*') if p.is_file()},**extra}))
    args=SimpleNamespace(cache_dir=cache,cohorts=cohort/'cohorts.csv',protocol=pp,panels=[sub/'proteins.csv'],output_dir=tmp_path/'out')
    return args,protocol


def test_actual_evaluation_does_not_aggregate_confirmation_rows(tmp_path):
    args,_=fixture(tmp_path);diag.evaluate(args)
    scores=pd.read_csv(args.output_dir/'scores.csv')
    assert len(scores)==4 and set(scores.record_id)=={'p0','p1','p2','p3'}
    assert set(scores.cohort)=={'generated_development','public_validation_development'}
    report=json.loads((args.output_dir/'manifest.json').read_text())
    assert report['scope']=='development_only_no_automatic_adoption_no_confirmation_read'


def test_protocol_cannot_relabel_confirmation_as_development(tmp_path):
    args,protocol=fixture(tmp_path);protocol['development_cohorts']=['public_test_confirmation']
    with pytest.raises(ValueError,match='Only the two'):diag.validate_frozen_design(args,protocol,{})


def test_changed_cohort_or_panel_file_is_rejected(tmp_path):
    args,protocol=fixture(tmp_path)
    args.cohorts.write_text(args.cohorts.read_text()+'\n')
    with pytest.raises(ValueError,match='Cohort file'):diag.validate_frozen_design(args,protocol,{})
