"""Parse and clean the fixed target protein FASTA."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_protein_sequence
from src.utils.fasta import read_fasta


def parse_targets(target_fasta: Path, output_csv: Path) -> pd.DataFrame:
    records = []
    for record in read_fasta(target_fasta):
        clean = clean_protein_sequence(record.sequence)
        records.append(
            {
                "target_id": record.record_id,
                "description": record.description,
                "target_sequence": clean.sequence,
                "length": len(clean.sequence),
                "is_valid": clean.is_valid,
                "clean_reason": clean.reason,
            }
        )
    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fasta", default="data/target/target.fa")
    parser.add_argument("--output", default="outputs/processed/targets.csv")
    args = parser.parse_args()
    df = parse_targets(Path(args.target_fasta), Path(args.output))
    print(f"targets: {len(df)} records -> {args.output}")
    print(df["is_valid"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

