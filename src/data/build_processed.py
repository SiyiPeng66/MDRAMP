"""Build all processed data files for the current repository data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.parse_private_round import parse_private_round
from src.data.parse_public import parse_public
from src.data.parse_targets import parse_targets


def build_processed(root: Path, *, round_name: str = "R0") -> dict[str, object]:
    output_dir = root / "outputs" / "processed"
    public = parse_public(root / "data" / "rawdata", output_dir)
    private = parse_private_round(
        root / "data" / "private" / f"{round_name}.txt",
        output_dir / f"private_{round_name}.csv",
        round_name=round_name,
    )
    targets = parse_targets(root / "data" / "target" / "target.fa", output_dir / "targets.csv")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "round": round_name,
        "public": {
            "raw_records": int(len(public["raw"])),
            "valid_cleaned_records": int(public["clean"]["is_valid"].sum()),
            "deduplicated_sequences": int(len(public["deduplicated"])),
            "clusters": int(public["split"]["cluster_id"].nunique()),
            "cluster_algorithm": "in_repository_length_aware_greedy_identity",
            "cluster_identity_threshold": 0.90,
        },
        "private": {
            "records": int(len(private)),
            "valid_records": int(private["is_valid"].sum()),
            "data_mode": "sequence_only",
        },
        "targets": {
            "records": int(len(targets)),
            "valid_records": int(targets["is_valid"].sum()),
        },
    }
    metadata_path = output_dir / "processed_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--round", default="R0")
    args = parser.parse_args()
    metadata = build_processed(Path(args.root).resolve(), round_name=args.round)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

