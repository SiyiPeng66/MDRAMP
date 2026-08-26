"""Post-campaign evidential calibration, separate from prospective ranking."""

from __future__ import annotations

import numpy as np


def beta_from_evidence(positive: np.ndarray, negative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(np.asarray(positive, dtype=float), 0.0)
    negative = np.maximum(np.asarray(negative, dtype=float), 0.0)
    return positive + 1.0, negative + 1.0


def evidential_summary(positive: np.ndarray, negative: np.ndarray) -> dict[str, np.ndarray]:
    alpha, beta = beta_from_evidence(positive, negative)
    total = alpha + beta
    vacuity = 2.0 / total
    reliability = 1.0 - vacuity
    return {
        "alpha": alpha,
        "beta": beta,
        "probability": alpha / total,
        "vacuity": vacuity,
        "reliability": reliability,
    }


def fuse_discounted_evidence(
    positive: np.ndarray,
    negative: np.ndarray,
) -> dict[str, np.ndarray]:
    """Discount each Expert's evidence by rho before summing across Experts.

    Inputs have shape ``(experts, candidates)``.
    """

    positive = np.maximum(np.asarray(positive, dtype=float), 0.0)
    negative = np.maximum(np.asarray(negative, dtype=float), 0.0)
    if positive.shape != negative.shape or positive.ndim != 2:
        raise ValueError("positive and negative evidence must be 2D with equal shape")
    rho = 1.0 - 2.0 / (positive + negative + 2.0)
    fused_positive = np.sum(rho * positive, axis=0)
    fused_negative = np.sum(rho * negative, axis=0)
    return evidential_summary(fused_positive, fused_negative)
