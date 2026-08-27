"""TPepPro score cache utilities and Noisy-OR aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PAIR_COLUMNS = {"peptide_id", "target_id", "s_aff_pair"}


def validate_pair_scores(pair_scores: pd.DataFrame) -> None:
    missing = REQUIRED_PAIR_COLUMNS - set(pair_scores.columns)
    if missing:
        raise ValueError(f"TPepPro pair score file is missing columns: {sorted(missing)}")
    if pair_scores["s_aff_pair"].isna().any():
        raise ValueError("TPepPro pair score file contains missing s_aff_pair values.")
    if ((pair_scores["s_aff_pair"] < 0) | (pair_scores["s_aff_pair"] > 1)).any():
        raise ValueError("TPepPro s_aff_pair values must be in [0, 1].")


def noisy_or(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - scores))


def aggregate_pair_frame(
    pair_scores: pd.DataFrame,
    *,
    top_k: int = 5,
    attention_noisy_or_weight: float,
) -> pd.DataFrame:
    if top_k != 5:
        raise ValueError("Paper mode fixes sparse interaction aggregation at top_k=5")
    if not 0.0 <= attention_noisy_or_weight <= 1.0:
        raise ValueError("attention_noisy_or_weight must be in [0, 1]")
    validate_pair_scores(pair_scores)
    rows = []
    for peptide_id, group in pair_scores.groupby("peptide_id", sort=True):
        sorted_group = group.sort_values("s_aff_pair", ascending=False)
        top = sorted_group.head(top_k)["s_aff_pair"].to_numpy(dtype=float)
        attention_logits = (
            sorted_group.head(top_k)["attention_logit"].to_numpy(dtype=float)
            if "attention_logit" in sorted_group else top
        )
        weights = np.exp(attention_logits - np.max(attention_logits))
        attention = float(np.sum((weights / weights.sum()) * top))
        coverage = noisy_or(group["s_aff_pair"].to_numpy())
        rows.append(
            {
                "peptide_id": peptide_id,
                "S_aff": attention_noisy_or_weight * attention + (1.0 - attention_noisy_or_weight) * coverage,
                "attention_top5": attention,
                "noisy_or": coverage,
                "aggregation_coefficient": attention_noisy_or_weight,
                "top_target_id": sorted_group.iloc[0]["target_id"],
                "top_target_score": float(sorted_group.iloc[0]["s_aff_pair"]),
                "target_count": int(len(group)),
                "teacher_model": "TPepPro",
            }
        )
    result = pd.DataFrame(rows)
    return result


def aggregate_pair_scores(
    pair_scores_csv: Path,
    output_csv: Path,
    *,
    top_k: int = 5,
    attention_noisy_or_weight: float,
) -> pd.DataFrame:
    result = aggregate_pair_frame(
        pd.read_csv(pair_scores_csv), top_k=top_k,
        attention_noisy_or_weight=attention_noisy_or_weight,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def write_actions_template(
    peptides_csv: Path,
    targets_csv: Path,
    output_tsv: Path,
    *,
    peptide_id_column: str = "private_record_id",
    peptide_sequence_column: str = "sequence",
    target_id_column: str = "target_id",
) -> pd.DataFrame:
    """Create a TPepPro pair action template.

    TPepPro preprocessing/model execution is kept external because it requires
    its own legacy environment and contact-map/embedding preparation. This
    template fixes the pair universe and downstream schema.
    """

    peptides = pd.read_csv(peptides_csv)
    targets = pd.read_csv(targets_csv)
    if "is_valid" in peptides.columns:
        peptides = peptides[peptides["is_valid"].astype(bool)]
    if "is_valid" in targets.columns:
        targets = targets[targets["is_valid"].astype(bool)]
    rows = []
    for pep in peptides.itertuples(index=False):
        peptide_id = getattr(pep, peptide_id_column)
        peptide_sequence = getattr(pep, peptide_sequence_column)
        for target in targets.itertuples(index=False):
            rows.append(
                {
                    "receptor_id": getattr(target, target_id_column),
                    "peptide_id": peptide_id,
                    "label": 0,
                    "peptide_sequence": peptide_sequence,
                }
            )
    result = pd.DataFrame(rows)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--pair-scores", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--aggregation-coefficient", type=float, required=True)

    template = sub.add_parser("template")
    template.add_argument("--peptides", required=True)
    template.add_argument("--targets", required=True)
    template.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "aggregate":
        df = aggregate_pair_scores(
            Path(args.pair_scores), Path(args.output),
            attention_noisy_or_weight=args.aggregation_coefficient,
        )
        print(f"TPepPro aggregated scores: {df.shape} -> {args.output}")
    elif args.command == "template":
        df = write_actions_template(Path(args.peptides), Path(args.targets), Path(args.output))
        print(f"TPepPro action template: {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()
