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


def amp_prior_loss(positive, unlabelled, *, positive_prior=0.10, manifold_weight=0.0):
    if manifold_weight != 0: raise ValueError("V6 has no manifold training loss")
    return non_negative_pu_loss(positive["pu"],unlabelled["pu"],positive_prior=positive_prior)


def masked_membrane_loss(outputs, primary_target, primary_mask, auxiliary_target=None, auxiliary_mask=None, *, teacher_weight=1.0):
    mask=primary_mask.bool() & torch.isfinite(primary_target) & (primary_target>=0) & (primary_target<=1) & ((primary_target<=.2)|(primary_target>=.8))
    if not mask.any(): return outputs["primary"].sum()*0
    return teacher_weight*torch.nn.functional.binary_cross_entropy(outputs["primary"][mask],primary_target[mask])
