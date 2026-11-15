import numpy as np
import torch
from src.models.expert_modules import PaperAMPPriorExpert, FeaturewiseGatedMembraneExpert
from src.train.expert_losses import masked_membrane_loss
from src.external.tpeppro_wrapper import aggregate_pair_frame
import pandas as pd


def test_support_calibration_and_frozen_branch():
    model=PaperAMPPriorExpert(40)
    z=torch.randn(15,8)
    model.fit_support(z,list(range(15)))
    output=model(torch.cat([z,torch.zeros(15,32)],1))
    assert torch.all(output['probability'] <= output['pu'])
    assert not model.reference.requires_grad
    assert not hasattr(model,'manifold_head')


def test_teacher_intermediate_and_invalid_zero_gradient():
    scores=torch.tensor([.4,.6,.7,.8],requires_grad=True)
    target=torch.tensor([.5,float('nan'),1.1,.9])
    loss=masked_membrane_loss({'primary':scores},target,torch.ones(4,dtype=torch.bool))
    loss.backward()
    assert torch.equal(scores.grad[:3],torch.zeros(3))
    assert scores.grad[3]!=0


def test_top_five_or_excludes_tail():
    frame=pd.DataFrame({'peptide_id':['x']*6,'target_id':list('abcdef'),'s_aff_pair':[.9,.8,.7,.6,.5,.4]})
    result=aggregate_pair_frame(frame,attention_noisy_or_weight=.8).iloc[0]
    assert np.isclose(result.noisy_or,1-np.prod(1-np.array([.9,.8,.7,.6,.5])))
