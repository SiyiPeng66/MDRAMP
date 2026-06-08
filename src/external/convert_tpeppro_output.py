"""Convert raw TPepPro predictions to MDRAMP pair-score schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_tpeppro_output(
    raw_csv: Path,
    output_csv: Path,
    *,
    receptor_column: str = "receptor",
    peptide_column: str = "peptide",
    score_column: str = "predict_score",
) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    for column in [receptor_column, peptide_column, score_column]:
        if column not in raw.columns:
            raise ValueError(f"Raw TPepPro output missing column: {column}")
    result = pd.DataFrame(
        {
            "peptide_id": raw[peptide_column].astype(str),
            "target_id": raw[receptor_column].astype(str),
            "s_aff_pair": raw[score_column].astype(float),
        }
    )
    if ((result["s_aff_pair"] < 0) | (result["s_aff_pair"] > 1)).any():
        raise ValueError("Converted TPepPro scores must be in [0, 1].")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receptor-column", default="receptor")
    parser.add_argument("--peptide-column", default="peptide")
    parser.add_argument("--score-column", default="predict_score")
    args = parser.parse_args()
    df = convert_tpeppro_output(
        Path(args.raw),
        Path(args.output),
        receptor_column=args.receptor_column,
        peptide_column=args.peptide_column,
        score_column=args.score_column,
    )
    print(f"converted TPepPro scores: {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()

