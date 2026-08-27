"""Paper-defined objectives for downstream MDRAMP Experts."""

from __future__ import annotations

import torch


def non_negative_pu_loss(
    positive_probability: torch.Tensor,
    unlabelled_probability: torch.Tensor,
    *,
    positive_prior: float = 0.5,
) -> torch.Tensor:
    """Non-negative PU risk using logistic loss and an operational background."""

    if not 0.0 < positive_prior < 1.0:
        raise ValueError("positive_prior must be in (0, 1)")
    eps = torch.finfo(positive_probability.dtype).eps
    pos = positive_probability.clamp(eps, 1.0 - eps)
    unl = unlabelled_probability.clamp(eps, 1.0 - eps)
    positive_risk = positive_prior * (-torch.log(pos)).mean()
    negative_risk = (-torch.log1p(-unl)).mean()
    positive_as_negative = positive_prior * (-torch.log1p(-pos)).mean()
    return positive_risk + torch.clamp(negative_risk - positive_as_negative, min=0.0)


def amp_prior_loss(
    positive: dict[str, torch.Tensor],
    unlabelled: dict[str, torch.Tensor],
    *,
    positive_prior: float = 0.5,
    manifold_weight: float = 1.0,
) -> torch.Tensor:
    pu = non_negative_pu_loss(
        positive["pu"], unlabelled["pu"], positive_prior=positive_prior
    )
    eps = torch.finfo(positive["manifold"].dtype).eps
    pos = positive["manifold"].clamp(eps, 1.0 - eps)
    unl = unlabelled["manifold"].clamp(eps, 1.0 - eps)
    manifold = 0.5 * ((-torch.log(pos)).mean() + (-torch.log1p(-unl)).mean())
    return pu + manifold_weight * manifold


def masked_membrane_loss(
    outputs: dict[str, torch.Tensor],
    primary_target: torch.Tensor,
    primary_mask: torch.Tensor,
    auxiliary_target: torch.Tensor,
    auxiliary_mask: torch.Tensor,
    *,
    teacher_weight: float = 1.0,
) -> torch.Tensor:
    """Continuous teacher BCE plus auxiliary BCE only where labels are visible."""

    bce = torch.nn.functional.binary_cross_entropy
    terms: list[torch.Tensor] = []
    if primary_mask.bool().any():
        terms.append(teacher_weight * bce(outputs["primary"][primary_mask], primary_target[primary_mask]))
    valid_aux = auxiliary_mask.bool()
    if valid_aux.any():
        terms.append(bce(outputs["auxiliary"][valid_aux], auxiliary_target[valid_aux]))
    if not terms:
        raise ValueError("Membrane batch contains no supervised labels")
    return torch.stack(terms).sum()
