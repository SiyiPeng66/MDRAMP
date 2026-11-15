import json,math
import pytest
from mdramp.calibration_screen_labels import assign_observation,SEMANTICS


def row(positive32='+',positive256='+',quant='',override=None):
    evidence=[{'cells':[{'header':'A. baumannii 32 μg/mL','value':positive32},{'header':'A. baumannii 256 μg/mL','value':positive256},{'header':'MIC_Ab_WT_μg_mL','value':quant}]}]
    if override is not None:evidence.append({'user_adjudication':{'mic_ug_ml':override}})
    return dict(provenance=json.dumps(evidence),lower_log2=-math.inf,upper_log2=5.,lower_inclusive=False,upper_inclusive=True,exact_log2=float('nan'),observation_type='left_censored')


def test_multiple_positive_doses_use_lowest_once():
    r=assign_observation(row());assert r['lower_log2']==r['upper_log2']==5
    assert r['target_semantics']==SEMANTICS and r['target_divisor']==1
    assert r['source_interval_lower_log2']==-math.inf
    assert r['label_assignment_source']=='user_assigned_lowest_positive_tested_dose'


def test_quantitative_measurement_has_priority_over_proxy():
    r=assign_observation(row(quant='2'));assert r['exact_log2']==1
    r=assign_observation(row('-', '+','32',32));assert r['exact_log2']==5
    assert r['label_assignment_source']=='user_adjudicated_quantitative_MIC'
    assert r['screen_quantitative_conflicts']=='negative_screen_inconsistent_with_quantitative_MIC'


def test_all_negative_remains_right_censored_and_nonmonotone_rejected():
    r=assign_observation(row('-','-'));assert r['lower_log2']==8 and r['upper_log2']==math.inf and not r['lower_inclusive']
    with pytest.raises(ValueError,match='Nonmonotone'):assign_observation(row('+','-'))


def test_256_positive_is_256_not_32_or_divided():
    r=assign_observation(row('-','+'));assert r['exact_log2']==8 and r['target_divisor']==1
