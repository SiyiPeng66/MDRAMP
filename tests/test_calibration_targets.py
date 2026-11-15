"""Design targets preserve original measurements and never imply assay improvement."""
import json
import math
import numpy as np
import pandas as pd
import pytest
from mdramp.calibration_targets import derive
from mdramp.calibration_train import sha


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir()
    obs=pd.DataFrame({'peptide_id':['a','b','c','d'],'strain_id':['MRSA']*4,
       'lower_log2':[5.,-np.inf,8.,5.],'upper_log2':[5.,8.,np.inf,8.],
       'exact_log2':[5.,np.nan,np.nan,np.nan],
       'lower_inclusive':[True,False,False,False],'upper_inclusive':[True,True,False,True],
       'observation_type':['exact','left_censored','right_censored','interval'],
       'status':['usable']*4,'cluster_id':['c0','c1','c2','c3'],'fold':[0,1,2,3],
       'provenance':['raw + / MIC32','raw +256','raw -256','raw -32 +256']})
    obs.to_csv(source/'observations.csv',index=False)
    pd.DataFrame({'peptide_id':obs.peptide_id,'sequence':['ACDEFGHIKL']*4,'cluster_id':obs.cluster_id,'fold':obs.fold}).to_csv(source/'sequences.csv',index=False)
    refresh(source)
    return source


def refresh(source):
    (source/'manifest.json').write_text(json.dumps({'schema_version':1,'output_sha256':{name:sha(source/name) for name in ['observations.csv','sequences.csv']}}))


def test_shift_bounds_preserves_measurements_groups_and_provenance(tmp_path):
    source=fixture(tmp_path);original={p.name:p.read_bytes() for p in source.iterdir()}
    out=tmp_path/'targets';derive(source,out,5.)
    a=pd.read_csv(source/'observations.csv');b=pd.read_csv(out/'observations.csv')
    for name in ['lower_log2','upper_log2','exact_log2']:
        np.testing.assert_allclose(b[name],a[name]-math.log2(5),equal_nan=True)
        np.testing.assert_allclose(b['measured_'+name],a[name],equal_nan=True)
    for name in ['lower_inclusive','upper_inclusive','status','cluster_id','fold','provenance','observation_type']:
        pd.testing.assert_series_equal(a[name],b[name])
    assert b.target_semantics.eq('design_target_not_measured').all() and b.target_divisor.eq(5).all()
    assert (out/'sequences.csv').read_bytes()==original['sequences.csv']
    assert all((source/name).read_bytes()==value for name,value in original.items())
    m=json.loads((out/'manifest.json').read_text());assert m['target_semantics']=='design_target_not_measured'
    assert all(sha(out/name)==value for name,value in m['output_sha256'].items())
    assert m['source_manifest_sha256']==sha(source/'manifest.json')
    with pytest.raises(ValueError,match='already derived'):derive(out,tmp_path/'twice',5)
    with pytest.raises(FileExistsError):derive(source,out,5)


@pytest.mark.parametrize('factor',[0,1,-5,float('nan'),float('inf')])
def test_invalid_divisor_fails_before_writes(tmp_path,factor):
    with pytest.raises(ValueError,match='divisor'):derive(tmp_path/'unused',tmp_path/'targets',factor)
    assert not (tmp_path/'targets').exists()


def test_changed_source_rejected(tmp_path):
    source=fixture(tmp_path);(source/'observations.csv').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):derive(source,tmp_path/'targets',5)
    assert not (tmp_path/'targets').exists()


def test_quarantine_never_rehabilitated_by_scaling(tmp_path):
    source=fixture(tmp_path);obs=pd.read_csv(source/'observations.csv');obs.loc[0,'status']='quarantined'
    obs.to_csv(source/'observations.csv',index=False);refresh(source)
    with pytest.raises(ValueError,match='Resolve'):derive(source,tmp_path/'targets',5)
    assert not (tmp_path/'targets').exists()
