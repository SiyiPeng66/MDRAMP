"""Exact support calibration and gradient equivalence for full-data training."""
import copy
import numpy as np
import pytest
import torch
from src.models.expert_modules import PaperAMPPriorExpert
from src.models.ensemble import ExpertEnsemble
from src.train.expert_losses import amp_prior_loss
from src.train.pretrain_public import fit_ensemble_support,train_amp_ensemble


def test_block_calibration_matches_original_leave_cluster_out():
    torch.manual_seed(71)
    raw=torch.randn(43,16)
    clusters=[i//3 for i in range(len(raw))]
    z=torch.nn.functional.normalize(raw,dim=1)
    old=torch.stack([(1-z[i]@z[torch.tensor([c!=clusters[i] for c in clusters])].T).topk(10,largest=False).values.mean() for i in range(len(z))])
    for block in [1,7,256]:
        model=PaperAMPPriorExpert(48)
        model.fit_support(raw,clusters,chunk_size=block)
        torch.testing.assert_close(model.radius,torch.quantile(old,.95),rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(model.spread,torch.quantile(old,.75)-torch.quantile(old,.25),rtol=1e-5,atol=1e-7)
        assert not model.reference.requires_grad
    with pytest.raises(ValueError):model.fit_support(raw,['same']*43)


def test_nnpu_discriminative_loss_and_gradients_match_full_forward():
    torch.manual_seed(4)
    model=PaperAMPPriorExpert(48)
    model.fit_support(torch.randn(20,16),list(range(20)))
    other=copy.deepcopy(model)
    positive=torch.randn(7,48);unlabelled=torch.randn(7,48)
    torch.manual_seed(91)
    full=amp_prior_loss(model(positive),model(unlabelled),positive_prior=.1)
    full.backward()
    torch.manual_seed(91)
    direct=amp_prior_loss(other.discriminative(positive),other.discriminative(unlabelled),positive_prior=.1)
    direct.backward()
    torch.testing.assert_close(full,direct,rtol=0,atol=0)
    for (name,p),(other_name,q) in zip(model.named_parameters(),other.named_parameters()):
        assert name==other_name
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_ensemble_calibrates_once_and_training_never_calls_support(monkeypatch):
    ensemble=ExpertEnsemble(lambda:PaperAMPPriorExpert(40),seeds=(11,23,37,51,73),output_key='probability')
    original=PaperAMPPriorExpert.fit_support;calls=[]
    def counted(self,*args,**kwargs):calls.append(self);return original(self,*args,**kwargs)
    monkeypatch.setattr(PaperAMPPriorExpert,'fit_support',counted)
    fit_ensemble_support(ensemble,torch.randn(15,8),list(range(15)))
    assert len(calls)==1
    for member in ensemble.members:
        for name in ['reference','reference_clusters','radius','spread']:
            torch.testing.assert_close(getattr(member,name),getattr(ensemble.members[0],name))
    def forbidden(*args,**kwargs):raise AssertionError('Training must not evaluate support')
    monkeypatch.setattr(PaperAMPPriorExpert,'forward',forbidden)
    values=np.random.default_rng(2).normal(size=(4,40)).astype(np.float32)
    history=train_amp_ensemble(ensemble,values,values[::-1].copy(),epochs=1,batch_size=4,learning_rate=1e-4,positive_prior=.1,manifold_weight=0,device='cpu')
    assert len(history)==5 and np.isfinite(history).all()
