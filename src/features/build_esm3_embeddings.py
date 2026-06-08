"""Build ESM3 embeddings from processed peptide or target CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.features.esm3_encoder import ESM3Encoder, ESM3SetupError, save_embeddings_npz


DATASET_CONFIG = {
    "public": {
        "input": "outputs/processed/public_cluster_split.csv",
        "id_column": "sequence",
        "sequence_column": "sequence",
        "valid_column": None,
        "prefix": "public_peptides",
    },
    "private_R0": {
        "input": "outputs/processed/private_R0.csv",
        "id_column": "private_record_id",
        "sequence_column": "sequence",
        "valid_column": "is_valid",
        "prefix": "private_R0_peptides",
    },
    "targets": {
        "input": "outputs/processed/targets.csv",
        "id_column": "target_id",
        "sequence_column": "target_sequence",
        "valid_column": "is_valid",
        "prefix": "targets",
    },
}


def load_records(
    input_csv: Path,
    *,
    id_column: str,
    sequence_column: str,
    valid_column: str | None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    df = pd.read_csv(input_csv)
    if valid_column and valid_column in df.columns:
        df = df[df[valid_column].astype(bool)].copy()
    df = df.dropna(subset=[id_column, sequence_column])
    if limit is not None:
        df = df.head(limit)
    return [
        (str(row[id_column]), str(row[sequence_column]))
        for _, row in df.iterrows()
    ]


def build_embeddings(
    *,
    dataset: str,
    input_csv: Path,
    output_dir: Path,
    id_column: str,
    sequence_column: str,
    valid_column: str | None,
    model_name: str,
    device: str,
    hf_home: Path,
    limit: int | None = None,
) -> None:
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str((hf_home / "hub").resolve())
    records = load_records(
        input_csv,
        id_column=id_column,
        sequence_column=sequence_column,
        valid_column=valid_column,
        limit=limit,
    )
    encoder = ESM3Encoder(model_name=model_name, device=device)
    embeddings = []
    for idx, (record_id, sequence) in enumerate(records, start=1):
        print(f"[{dataset}] ESM3 encoding {idx}/{len(records)}: {record_id}")
        embeddings.append(encoder.encode_sequence(record_id, sequence))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_embeddings_npz(
        embeddings,
        output_npz=output_dir / f"{dataset}.npz",
        metadata_csv=output_dir / f"{dataset}_metadata.csv",
        metadata_json=output_dir / f"{dataset}_metadata.json",
        model_name=model_name,
    )
    print(f"ESM3 embeddings written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIG),
        required=True,
        help="Processed dataset to embed.",
    )
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default="outputs/esm3_embeddings")
    parser.add_argument("--model-name", default="esm3-sm-open-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-home", default="outputs/hf_cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Only verify that the required ESM3 package can be imported.",
    )
    args = parser.parse_args()

    if args.check_dependencies:
        os.environ["HF_HOME"] = str(Path(args.hf_home).resolve())
        os.environ["HUGGINGFACE_HUB_CACHE"] = str((Path(args.hf_home) / "hub").resolve())
        try:
            ESM3Encoder(model_name=args.model_name, device=args.device)
        except ESM3SetupError as exc:
            raise SystemExit(str(exc)) from exc
        print("ESM3 dependency and checkpoint are available.")
        return

    config = DATASET_CONFIG[args.dataset]
    build_embeddings(
        dataset=config["prefix"],
        input_csv=Path(args.input_csv or config["input"]),
        output_dir=Path(args.output_dir),
        id_column=config["id_column"],
        sequence_column=config["sequence_column"],
        valid_column=config["valid_column"],
        model_name=args.model_name,
        device=args.device,
        hf_home=Path(args.hf_home),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
