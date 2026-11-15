import pandas as pd
import pytest
from src.selection.pareto import measured_pareto_front


def test_measured_selection_compares_all_strains():
    frame=pd.DataFrame({'id':['a','b','c'],'mic1':[2,4,8],'mic2':[8,2,16],'MHC5':[128,128,64]})
    assert measured_pareto_front(frame,mic_columns=['mic1','mic2'])['id'].tolist()==['a','b']


def test_censored_endpoints_not_treated_as_exact():
    frame=pd.DataFrame({'mic':[2],'MHC5':[512],'MHC5_censored':[True]})
    assert len(measured_pareto_front(frame,mic_columns=['mic'])) == 1


def test_local_freeze_does_not_apply_de_novo_diversity():
    import numpy as np
    from src.selection.freeze_round import freeze_round_selection
    parent='KALWQKALWQKALWQKALWQ'
    variant='RALWQKALWQKALWQKALWQ'
    frame=pd.DataFrame({'sequence':[variant]*200,'R_t':[.9]*200,'selection_category':['exploitation']*200})
    selected,reserve=freeze_round_selection(frame,np.zeros((200,3)),budget=1,reserve_size=0,quotas={'exploitation':1},reference_sequences=[parent],parent=parent,parent_family=[parent])
    assert selected.sequence.tolist()==[variant]
    with pytest.raises(ValueError,match='include parent'):
        freeze_round_selection(frame,np.zeros((200,3)),budget=1,reserve_size=0,quotas={'exploitation':1},parent=parent)
