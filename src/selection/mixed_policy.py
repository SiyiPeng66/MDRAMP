"""Deterministic mixed candidate selection with frozen quotas."""

from __future__ import annotations

import pandas as pd


DEFAULT_QUOTAS = {"exploitation": 0.60, "uncertainty": 0.15, "diversity": 0.15, "novelty": 0.10}


def select_mixed(
    candidates: pd.DataFrame,
    *,
    budget: int,
    score_column: str = "R_t",
    category_column: str = "selection_category",
    quotas: dict[str, float] = DEFAULT_QUOTAS,
) -> pd.DataFrame:
    if score_column not in candidates.columns:
        raise ValueError(f"Missing ranking score: {score_column}")
    remaining = candidates.copy()
    selected = []
    for category, fraction in quotas.items():
        count = int(round(budget * fraction))
        pool = remaining[remaining[category_column] == category].sort_values(score_column, ascending=False)
        take = pool.head(count)
        selected.append(take)
        remaining = remaining.drop(take.index)
    result = pd.concat(selected, ignore_index=True) if selected else remaining.head(0)
    if len(result) < budget:
        result = pd.concat([result, remaining.sort_values(score_column, ascending=False).head(budget - len(result))], ignore_index=True)
    return result.head(budget).reset_index(drop=True)
