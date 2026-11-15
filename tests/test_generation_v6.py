import torch
from src.generation.policy import Policy, reinforce_loss

def test_masked_policy_and_conditional_loss():
    torch.manual_seed(1)
    model=Policy(hidden=16,heads=2,layers=1,feedforward=32,dropout=0.)
    model.eval()
    seq='ACDEFGHIKL'
    logp=model.log_probability(seq)
    assert torch.isfinite(logp)
    loss=model.supervised_loss([seq], [seq])
    loss.backward()
    assert torch.isfinite(loss)
    legal=model.legal_logits(torch.zeros(24),0)
    assert torch.isneginf(legal[22])
    terminal=model.legal_logits(torch.zeros(24),50).softmax(-1)
    assert terminal[22]==1

def test_reinforce_single_zero_advantage():
    model=Policy(hidden=16,heads=2,layers=1,feedforward=32,dropout=0.)
    model.eval()
    loss=reinforce_loss(model,model,['ACDEFGHIKL'],[0.],.1)
    assert loss.item()==0

def test_rl_requires_explicit_optimizer_and_reference_inputs(tmp_path, monkeypatch):
    import sys
    import pytest
    from src.generation.cli import save, main
    model=Policy(hidden=16,heads=2,layers=1,feedforward=32,dropout=0.)
    path=tmp_path/'model.pt'; save(model,path,False)
    args=['generation','reinforce','--checkpoint',str(path),'--reference',str(path),
          '--output',str(tmp_path/'out.pt'),'--seed','1','--scorer-config','unused.json',
          '--steps','1','--beta-kl','.1']
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(SystemExit): main()
    monkeypatch.setattr(sys,'argv',args+['--learning-rate','.0001','--batch-size','2','--weight-decay','0'])
    with pytest.raises(SystemExit): main()
