"""Parse private round data such as data/private/R0.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_peptide_sequence


def parse_private_round(
    input_path: Path,
    output_csv: Path,
    *,
    round_name: str,
) -> pd.DataFrame:
    rows = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            value = raw_line.strip()
            if not value:
                continue
            if value.startswith(">"):
                continue
            clean = clean_peptide_sequence(value)
            rows.append(
                {
                    "round": round_name,
                    "private_record_id": f"{round_name}_{idx:05d}",
                    "raw_sequence": value,
                    "sequence": clean.sequence,
                    "length": len(clean.sequence),
                    "is_valid": clean.is_valid,
                    "clean_reason": clean.reason,
                    "data_mode": "sequence_only",
                }
            )
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/private/R0.txt")
    parser.add_argument("--round", default="R0")
    parser.add_argument("--output", default="outputs/processed/private_R0.csv")
    args = parser.parse_args()
    df = parse_private_round(Path(args.input), Path(args.output), round_name=args.round)
    print(f"private {args.round}: {len(df)} records -> {args.output}")
    print(df["is_valid"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

