import numpy as np
import pandas as pd
import pytest
from mdramp.prospective_scenarios import apply_scenario

def fixture():
    frame=pd.DataFrame({'sequence':['AAA','CCC','DDD'],'round':['R3','R0',None],'p_cons':[.9,.8,.7],'predicted_adjusted_log2_mic':[3.,4.,5.],'target_divisor':[10.,10.,10.]})
    policy={'default_divisor':10.,'round_divisors':{'R3':15.},'sequence_divisors':{'AAA':20.}}
    return frame,policy

def test_sequence_override_and_untouched_rows():
    f,p=fixture();r=apply_scenario(f,p)
    assert r.scenario_predicted_adjusted_mic_ug_ml.tolist()==[4.,16.,32.]
    assert r.scenario_rank.iloc[0]==1 and pd.isna(r.scenario_rank.iloc[2])
    assert r.p_cons.equals(f.p_cons)

def test_reject_missing_scores_and_wrong_base():
    f,p=fixture();f.loc[0,'p_cons']=np.nan
    with pytest.raises(ValueError,match='finite'):apply_scenario(f,p)
    f,p=fixture();f.loc[0,'target_divisor']=5
    with pytest.raises(ValueError,match='baseline'):apply_scenario(f,p)
