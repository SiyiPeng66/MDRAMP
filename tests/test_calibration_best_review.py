"""Independent minimum-interval boundary and target-scale review cases."""
import math
import numpy as np
import pandas as pd
import pytest
from mdramp.calibration_best import minimum_interval, aggregate, STRAINS


def interval(lo,hi,li=True,ui=True):
    return dict(lower_log2=lo,upper_log2=hi,lower_inclusive=li,upper_inclusive=ui)


@pytest.mark.parametrize('rows,expected',[
    ([interval(2,5,False,True),interval(2,6)],(2,5,True,True)),
    ([interval(1,5,True,False),interval(2,5)],(1,5,True,False)),
    ([interval(5,5),interval(5,math.inf,False,False)],(5,5,True,True)),
    ([interval(5,5),interval(-math.inf,8,False,True)],(-math.inf,5,False,True)),
    ([interval(8,math.inf,False,False),interval(8,math.inf,False,False)],(8,math.inf,False,False)),
])
def test_minimum_interval_ties_and_censoring(rows,expected):
    assert minimum_interval(rows)==expected
    shift=math.log2(5)
    shifted=[{**r,'lower_log2':r['lower_log2']-shift,'upper_log2':r['upper_log2']-shift} for r in rows]
    actual=minimum_interval(shifted)
    np.testing.assert_allclose(actual[:2],[expected[0]-shift,expected[1]-shift])
    assert actual[2:]==expected[2:]


def fixture():
    seq=pd.DataFrame([{'peptide_id':'p1','sequence':'ACDEFGHIKL','cluster_id':'c1','fold':3}])
    obs=pd.DataFrame([{'peptide_id':'p1','strain_id':s,'status':'usable','cluster_id':'c1','fold':3,
       **interval(8,math.inf,False,False),'provenance':'source'} for s in STRAINS])
    obs.loc[0,['lower_log2','upper_log2','lower_inclusive','upper_inclusive']]=[5,5,True,True]
    return seq,obs


def test_one_row_per_peptide_no_fold_or_source_mutation():
    seq,obs=fixture();before=obs.copy(deep=True)
    result=aggregate(seq,obs)
    assert len(result)==1 and result.iloc[0].strain_id=='best_panel'
    assert result.iloc[0].exact_log2==5 and result.iloc[0].fold==3
    assert result.iloc[0].cluster_id=='c1' and result.iloc[0].source_strain_count==6
    pd.testing.assert_frame_equal(obs,before)


@pytest.mark.parametrize('corruption',['missing','quarantined','fold','duplicate_strain'])
def test_incomplete_or_inconsistent_panels_cannot_be_silently_reduced(corruption):
    seq,obs=fixture()
    if corruption=='missing':obs=obs.iloc[:-1]
    if corruption=='quarantined':obs.loc[0,'status']='quarantined'
    if corruption=='fold':obs.loc[0,'fold']=4
    if corruption=='duplicate_strain':obs.loc[0,'strain_id']=STRAINS[1]
    with pytest.raises(ValueError):aggregate(seq,obs)


def test_derived_six_strain_targets_cannot_be_relabelled_as_measurements():
    seq,obs=fixture();obs['target_semantics']='design_target_not_measured';obs['target_divisor']=5
    with pytest.raises(ValueError):aggregate(seq,obs)
