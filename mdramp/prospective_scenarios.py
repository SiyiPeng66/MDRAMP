"""Explore transparent MIC divisor scenarios without changing frozen model weights.

The base prediction is the final575 /10 head output.  A scenario divisor ``d``
represents the explicit counterfactual MIC rule ``MIC = raw_MIC / d``; hence
the adjusted log2 prediction is shifted by ``log2(10 / d)``.  Experts,
protein panel, and candidate sequences are unchanged.  This is a design
exploration artifact, not a retrained or experimentally validated model.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("scenario policy schema_version must be 1")
    default = float(policy.get("default_divisor", 10))
    if not math.isfinite(default) or default <= 0:
        raise ValueError("default_divisor must be positive")
    rounds = {str(k): float(v) for k, v in policy.get("round_divisors", {}).items()}
    sequences = {str(k): float(v) for k, v in policy.get("sequence_divisors", {}).items()}
    if any(v <= 0 or not math.isfinite(v) for v in [*rounds.values(), *sequences.values()]):
        raise ValueError("all scenario divisors must be positive finite values")
    return {"schema_version": 1, "default_divisor": default,
            "round_divisors": rounds, "sequence_divisors": sequences,
            "name": str(policy.get("name", "unnamed")),
            "rationale": str(policy.get("rationale", ""))}


def apply_scenario(frame: pd.DataFrame, policy: dict) -> pd.DataFrame:
    if 'target_divisor' not in frame or not frame.target_divisor.eq(10).all():
        raise ValueError("Input must explicitly use baseline target_divisor=10")
    if frame.sequence.isna().any() or frame.sequence.astype(str).str.strip().eq('').any():
        raise ValueError("Sequence must not be empty")
    numeric = frame[["p_cons", "predicted_adjusted_log2_mic"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Complete finite scores required; fill missing frozen expert scores first")
    if not frame.p_cons.between(0, 1).all():
        raise ValueError("p_cons must lie in [0,1]")
    out = frame.copy()
    divisors = []
    for row in out.itertuples():
        divisors.append(policy["sequence_divisors"].get(row.sequence,
                        policy["round_divisors"].get(row.round, policy["default_divisor"])))
    out["scenario_divisor"] = np.asarray(divisors, dtype=float)
    out["scenario_log2_shift"] = np.log2(10.0 / out.scenario_divisor)
    out["scenario_predicted_adjusted_log2_mic"] = out.predicted_adjusted_log2_mic + out.scenario_log2_shift
    out["scenario_predicted_adjusted_mic_ug_ml"] = np.exp2(out.scenario_predicted_adjusted_log2_mic)
    out["scenario_A_t"] = 1.0 / (1.0 + np.exp(out.scenario_predicted_adjusted_log2_mic - 5.0))
    out["scenario_R_t"] = out.p_cons * out.scenario_A_t
    if not np.isfinite(out[["scenario_predicted_adjusted_mic_ug_ml", "scenario_R_t"]].to_numpy(float)).all():
        raise ValueError("Scenario produced nonfinite values")
    out["scenario_rank"] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    for rnd, idx in out.groupby("round").groups.items():
        ordered = out.loc[idx].sort_values(["scenario_R_t", "sequence"], ascending=[False, True]).index
        out.loc[ordered, "scenario_rank"] = np.arange(1, len(ordered) + 1)
    return out


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rnd, part in frame.groupby("round", sort=True):
        vals = part.scenario_predicted_adjusted_mic_ug_ml.to_numpy(float)
        rows.append({"round": rnd, "n": len(part), "divisor_min": part.scenario_divisor.min(),
                     "divisor_max": part.scenario_divisor.max(), "mic_median": np.median(vals),
                     "mic_q25": np.quantile(vals, .25), "mic_q75": np.quantile(vals, .75),
                     "mic_min": vals.min(), "mic_max": vals.max(),
                     "fraction_le_128": float((vals <= 128).mean())})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--policy", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(argv); source = Path(a.input).resolve(); policy_path = Path(a.policy).resolve()
    frame = pd.read_csv(source)
    if 'target_semantics' not in frame or not frame.target_semantics.eq('user_adjusted_approximate_mic').all():
        raise ValueError('Expected explicit user-adjusted MIC inputs')
    if 'state_id' not in frame or frame.state_id.isna().any() or frame.state_id.nunique() != 1:
        raise ValueError('One explicit baseline state_id is required')
    required = {"round", "sequence", "p_cons", "predicted_adjusted_log2_mic"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"input missing columns: {sorted(missing)}")
    if frame.sequence.duplicated().any(): raise ValueError("input sequences must be unique")
    policy = _load_policy(policy_path)
    if set(policy['sequence_divisors']) - set(frame.sequence):
        raise ValueError('Policy references absent sequences')
    if set(policy['round_divisors']) - set(frame['round'].dropna()):
        raise ValueError('Policy references absent rounds')
    result = apply_scenario(frame, policy)
    out = Path(a.output_dir).resolve()
    if out.exists(): raise FileExistsError('Use a new output directory for each scenario run')
    out.mkdir(parents=True)
    result.to_csv(out / "scenario_scores.csv", index=False)
    summarize(result).to_csv(out / "scenario_round_summary.csv", index=False)
    selected = result[result.sequence.isin(policy["sequence_divisors"])]
    selected.to_csv(out / "scenario_selected.csv", index=False)
    manifest = {"status": "complete", "schema_version": 1, "scenario": policy,
                "input": str(source), "input_sha256": _sha(source),
                "policy_sha256": _sha(policy_path), "record_count": len(result),
                "baseline_state_id": str(frame.state_id.iloc[0]),
                "frozen_weights_changed": False,
                "interpretation": "counterfactual divisor/post-calibration exploration; no retraining or validation claim",
                "outputs": {x.name: _sha(x) for x in out.glob("*.csv")}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"output_dir": str(out), "record_count": len(result), "scenario": policy["name"]}, ensure_ascii=False))


if __name__ == "__main__": main()
