"""Post-campaign calibration metrics."""

from __future__ import annotations

import numpy as np


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability) - np.asarray(y_true)) ** 2))


def negative_log_likelihood(y_true: np.ndarray, probability: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & ((p < right) if right < 1.0 else (p <= right))
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)
