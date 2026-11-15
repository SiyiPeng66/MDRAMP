from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from mdramp.calibration_train import CalibrationMLP
from mdramp.release_calibration import train_head


def settings(**kwargs):
    return SimpleNamespace(seed=11,epochs=20,batch_size=3,learning_rate=.001,**kwargs)


def test_real_warm_start_updates_shared_weights_and_preserves_parent():
    torch.manual_seed(4)
    model=CalibrationMLP(1571,16)
    before={k:v.clone() for k,v in model.state_dict().items()}
    x=np.random.default_rng(3).normal(size=(3,1571)).astype('float32')
    best=pd.DataFrame({'lower_log2':[-1.,0.,1.],'upper_log2':[-1.,0.,1.]})
    trained,pred_before,pred_after,history=train_head(x,best,model,settings())
    assert any(not torch.equal(v,trained.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert np.isfinite(pred_after).all() and not np.array_equal(pred_before,pred_after)
    assert history[-1]['interval_huber']<history[0]['interval_huber']


def test_base_initialization_and_censoring_are_finite_and_deterministic():
    x=np.random.default_rng(9).normal(size=(2,1571)).astype('float32')
    best=pd.DataFrame({'lower_log2':[1.,4.],'upper_log2':[1.,np.inf]})
    a,b,p,_=train_head(x,best,None,settings())
    assert b is None and np.isfinite(p).all()
    a2,_,p2,_=train_head(x,best,None,settings())
    assert np.array_equal(p,p2)
    assert all(torch.equal(v,a2.state_dict()[k]) for k,v in a.state_dict().items())
