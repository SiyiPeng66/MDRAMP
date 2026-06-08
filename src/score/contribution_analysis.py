"""Post-hoc normalized expert contribution analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.gated_integrator import GateParams, integrate_numpy


def contribution_table(scores_csv: Path, output_csv: Path, params: GateParams) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    base = df["p_cons"].to_numpy()
    baselines = {
        "prior": float(df["S_prior"].mean()),
        "membrane": float(df["S_mem"].mean()),
        "target_affinity": float(df["S_aff"].mean()),
    }
    deltas = {}
    for key, column in {
        "prior": "S_prior",
        "membrane": "S_mem",
        "target_affinity": "S_aff",
    }.items():
        S_prior = df["S_prior"].to_numpy().copy()
        S_mem = df["S_mem"].to_numpy().copy()
        S_aff = df["S_aff"].to_numpy().copy()
        if column == "S_prior":
            S_prior[:] = baselines[key]
        elif column == "S_mem":
            S_mem[:] = baselines[key]
        else:
            S_aff[:] = baselines[key]
        recomputed = integrate_numpy(S_prior, S_mem, S_aff, params)["p_cons"]
        deltas[key] = float(np.mean(np.abs(base - recomputed)))
    denom = sum(deltas.values()) or 1.0
    result = pd.DataFrame(
        [
            {"expert": key, "mean_abs_delta": value, "normalized_contribution": value / denom}
            for key, value in deltas.items()
        ]
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", default="outputs/scores/contribution_analysis.csv")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--lambda-syn", type=float, default=0.3)
    args = parser.parse_args()
    df = contribution_table(
        Path(args.scores),
        Path(args.output),
        GateParams(args.tau, args.gamma, args.lambda_syn),
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

