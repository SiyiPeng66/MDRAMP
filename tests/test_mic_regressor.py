import torch

from src.models.mic_regressor import (
    MICRegressionHead,
    activity_priority,
    censored_huber_loss,
)


def test_mic_head_forward_and_activity_mapping():
    model = MICRegressionHead(peptide_dim=8, physchem_dim=4)
    prediction = model(torch.zeros(2, 8), torch.zeros(2, 4), torch.zeros(2, 3))
    assert prediction.shape == (2,)
    assert torch.isclose(activity_priority(torch.tensor([5.0]))[0], torch.tensor(0.5))


def test_censored_loss_does_not_penalize_prediction_above_limit():
    prediction = torch.tensor([9.0, 7.0, 4.0])
    target = torch.tensor([8.0, 8.0, 5.0])
    censored = torch.tensor([True, True, False])
    loss = censored_huber_loss(prediction, target, censored)
    expected = (0.0 + 0.5 + 0.5) / 3.0
    assert torch.isclose(loss, torch.tensor(expected))


def test_mic_expert_features_are_detached():
    model = MICRegressionHead(peptide_dim=2, physchem_dim=1)
    peptide = torch.zeros(1, 2, requires_grad=True)
    physchem = torch.zeros(1, 1, requires_grad=True)
    means = torch.zeros(1, 3, requires_grad=True)
    model(peptide, physchem, means).sum().backward()
    assert means.grad is None
