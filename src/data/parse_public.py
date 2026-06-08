"""Parse, clean, deduplicate, cluster, and split public AMP raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_peptide_sequence
from src.data.cluster_split import assign_cluster_splits, greedy_cluster_sequences
from src.utils.fasta import read_fasta


FASTA_EXTENSIONS = {".fa", ".fasta", ".faa"}


def source_from_path(path: Path) -> str:
    parts = path.parts
    if "rawdata" in parts:
        idx = parts.index("rawdata")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def parse_fasta_like(path: Path) -> list[dict[str, str]]:
    source = source_from_path(path)
    rows = []
    for record in read_fasta(path):
        rows.append(
            {
                "source": source,
                "source_file": str(path),
                "source_record_id": record.record_id,
                "description": record.description,
                "raw_sequence": record.sequence,
            }
        )
    return rows


def parse_camp_csv(path: Path) -> list[dict[str, str]]:
    source = source_from_path(path)
    df = pd.read_csv(path, encoding="latin1")
    sequence_col = "Seqence" if "Seqence" in df.columns else "Sequence"
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "source": source,
                "source_file": str(path),
                "source_record_id": str(row.get("Camp_ID", "")),
                "description": str(row.get("Title", "")),
                "raw_sequence": str(row.get(sequence_col, "")),
            }
        )
    return rows


def parse_public_rawdata(rawdata_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in sorted(rawdata_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in FASTA_EXTENSIONS:
            rows.extend(parse_fasta_like(path))
        elif suffix == ".txt":
            rows.extend(parse_fasta_like(path))
        elif suffix == ".csv":
            rows.extend(parse_camp_csv(path))
        elif suffix == ".xlsx":
            continue
    return pd.DataFrame(rows)


def clean_and_deduplicate(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_rows = []
    for idx, row in raw_df.iterrows():
        clean = clean_peptide_sequence(row["raw_sequence"])
        cleaned_rows.append(
            {
                **row.to_dict(),
                "sequence": clean.sequence,
                "length": len(clean.sequence),
                "is_valid": clean.is_valid,
                "clean_reason": clean.reason,
            }
        )
    clean_df = pd.DataFrame(cleaned_rows)
    valid_df = clean_df[clean_df["is_valid"]].copy()

    grouped = (
        valid_df.groupby("sequence", as_index=False)
        .agg(
            length=("length", "first"),
            sources=("source", lambda values: ";".join(sorted(set(map(str, values))))),
            source_record_ids=(
                "source_record_id",
                lambda values: ";".join(sorted(set(map(str, values)))[:20]),
            ),
            source_count=("source", "count"),
        )
        .sort_values(["length", "sequence"], ignore_index=True)
    )
    grouped["label_amp_prior"] = 1
    return clean_df, grouped


def add_cluster_split(
    dedup_df: pd.DataFrame,
    *,
    identity_threshold: float = 0.90,
) -> pd.DataFrame:
    assignments = greedy_cluster_sequences(
        dedup_df["sequence"].tolist(), identity_threshold=identity_threshold
    )
    cluster_df = pd.DataFrame([assignment.__dict__ for assignment in assignments])
    result = dedup_df.merge(cluster_df, on="sequence", how="left")
    split_map = assign_cluster_splits(result["cluster_id"].tolist())
    result["split"] = result["cluster_id"].map(split_map)
    return result.sort_values(["split", "cluster_id", "sequence"], ignore_index=True)


def parse_public(
    rawdata_dir: Path,
    output_dir: Path,
    *,
    identity_threshold: float = 0.90,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = parse_public_rawdata(rawdata_dir)
    clean_df, dedup_df = clean_and_deduplicate(raw_df)
    split_df = add_cluster_split(dedup_df, identity_threshold=identity_threshold)

    raw_df.to_csv(output_dir / "public_raw_records.csv", index=False)
    clean_df.to_csv(output_dir / "public_clean_records.csv", index=False)
    dedup_df.to_csv(output_dir / "public_deduplicated.csv", index=False)
    split_df.to_csv(output_dir / "public_cluster_split.csv", index=False)

    return {
        "raw": raw_df,
        "clean": clean_df,
        "deduplicated": dedup_df,
        "split": split_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rawdata-dir", default="data/rawdata")
    parser.add_argument("--output-dir", default="outputs/processed")
    parser.add_argument("--identity-threshold", type=float, default=0.90)
    args = parser.parse_args()

    dfs = parse_public(
        Path(args.rawdata_dir),
        Path(args.output_dir),
        identity_threshold=args.identity_threshold,
    )
    print(f"public raw records: {len(dfs['raw'])}")
    print(f"public valid cleaned records: {int(dfs['clean']['is_valid'].sum())}")
    print(f"public deduplicated sequences: {len(dfs['deduplicated'])}")
    print(f"public clusters: {dfs['split']['cluster_id'].nunique()}")
    print(f"outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
