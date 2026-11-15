"""Independent manuscript S3/S10 probability and gradient checks."""
import math
import torch
from src.generation.policy import Policy,AA,EOS,reinforce_loss


class ConstantPolicy(Policy):
    def __init__(self):
        torch.nn.Module.__init__(self);self.weight=torch.nn.Parameter(torch.zeros(24))
    def forward(self,tokens):return self.weight.expand(tokens.shape[0],tokens.shape[1],24)


def test_sequence_probability_includes_eos_and_masks_special_actions():
    policy=ConstantPolicy()
    for length in [10,15,50]:
        # First 10 actions have 20 residues; later actions have 20 residues+EOS.
        # At length 50 the forced EOS has log probability zero.
        expected=-10*math.log(20)-(length-10+(length<50))*math.log(21)
        assert torch.isclose(policy.log_probability('A'*length),torch.tensor(expected),atol=2e-5)
        assert torch.isclose(policy.log_probability('A'*length,parent='C'*10),torch.tensor(expected),atol=2e-5)
    assert torch.isneginf(policy.legal_logits(torch.zeros(24),0)[EOS])
    legal=policy.legal_logits(torch.zeros(24),50)
    assert torch.isfinite(legal).sum()==1 and legal[EOS]==0


def test_conditional_supervision_excludes_parent_and_padding():
    model=ConstantPolicy()
    seq=['A'*10,'C'*15];parents=['D'*50,'E'*10]
    loss=model.supervised_loss(seq,parents)
    assert torch.isclose(loss,torch.tensor(math.log(24)),atol=1e-6)
    loss.backward()
    # Only target variants + EOS are supervised (27 total tokens).
    expected=torch.full((24,),1/24.)
    expected[AA.index('A')]-=10/27;expected[AA.index('C')]-=15/27;expected[EOS]-=2/27
    torch.testing.assert_close(model.weight.grad,expected)


def test_loo_kl_return_gradient_keeps_zero_reward_trajectory():
    class LogPolicy(torch.nn.Module):
        def __init__(self,values):super().__init__();self.values=torch.nn.Parameter(torch.tensor(values))
        def log_probability(self,sequence,parent=None):return self.values[int(sequence)]
    policy=LogPolicy([-2.,-3.]);reference=LogPolicy([-2.5,-2.])
    loss=reinforce_loss(policy,reference,['0','1'],[1.,0.],.2)
    loss.backward()
    # G=[.9,.2], leave-one-out baseline=[.2,.9]; both sampled paths contribute.
    torch.testing.assert_close(policy.values.grad,torch.tensor([-.35,.35]))
    assert reference.values.grad is None
