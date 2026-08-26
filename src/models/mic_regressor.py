"""Lightweight quantitative MIC head used by the prospective model states."""

from __future__ import annotations

import torch


class MICRegressionHead(torch.nn.Module):
    """Predict log2(MIC) from frozen representation and Expert evidence means."""

    def __init__(
        self,
        peptide_dim: int = 1536,
        physchem_dim: int = 32,
        expert_dim: int = 3,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.peptide_dim = peptide_dim
        self.physchem_dim = physchem_dim
        self.expert_dim = expert_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(peptide_dim + physchem_dim + expert_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        peptide_embedding: torch.Tensor,
        physchem: torch.Tensor,
        expert_means: torch.Tensor,
    ) -> torch.Tensor:
        if expert_means.shape[-1] != self.expert_dim:
            raise ValueError(
                f"Expected {self.expert_dim} Expert means, got {expert_means.shape[-1]}."
            )
        # Expert evidence is deliberately detached: MIC feedback updates only
        # this lightweight head, never the biologically supervised Experts.
        features = torch.cat(
            [peptide_embedding, physchem, expert_means.detach()], dim=-1
        )
        return self.net(features).squeeze(-1)


def huber_loss(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Elementwise Huber loss for log2(MIC) residuals."""

    absolute = residual.abs()
    quadratic = torch.minimum(absolute, torch.tensor(delta, device=residual.device))
    linear = absolute - quadratic
    return 0.5 * quadratic.square() + delta * linear


def censored_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    censored: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss with one-sided right-censoring for upper MIC limits.

    For a censored observation, only predictions below the censoring threshold
    are penalized: ``residual = max(0, target - prediction)``.
    """

    if prediction.shape != target.shape or prediction.shape != censored.shape:
        raise ValueError("prediction, target and censored must have identical shapes")
    ordinary_residual = prediction - target
    censored_residual = torch.clamp(target - prediction, min=0.0)
    residual = torch.where(censored.bool(), censored_residual, ordinary_residual)
    return huber_loss(residual, delta=delta).mean()


def activity_priority(predicted_log2_mic: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Map predicted log2(MIC) to the fixed activity-priority signal A_t."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid((5.0 - predicted_log2_mic) / temperature)
