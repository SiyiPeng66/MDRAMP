"""Public-data initialization for M_R0."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.physicochemical import add_physicochemical_features
from src.models.expert_modules import AMPPriorExpert, MembraneMechanismExpert
from src.train.train_utils import load_peptide_matrix, require_file, save_checkpoint, save_json, train_binary_regressor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default="outputs/processed/public_cluster_split.csv")
    parser.add_argument("--features", default="outputs/processed/public_features.csv")
    parser.add_argument("--embeddings", default="outputs/esm3_embeddings/public_peptides.npz")
    parser.add_argument("--embedding-metadata", default="outputs/esm3_embeddings/public_peptides_metadata.csv")
    parser.add_argument("--pore-scores", default="outputs/teacher_scores/pore_public.csv")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/M_R0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    add_physicochemical_features(Path(args.processed), Path(args.features), sequence_column="sequence")
    df, matrix = load_peptide_matrix(
        Path(args.features),
        Path(args.embeddings),
        Path(args.embedding_metadata),
        id_column="sequence",
    )

    amp = AMPPriorExpert(input_dim=matrix.shape[1])
    amp_losses = train_binary_regressor(
        amp,
        matrix,
        df["label_amp_prior"].to_numpy(dtype="float32"),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_checkpoint(
        amp,
        Path(args.checkpoint_dir) / "amp_prior.pt",
        metadata={"input_dim": matrix.shape[1], "losses": amp_losses, "source": "public"},
    )

    pore_scores = pd.read_csv(require_file(args.pore_scores))
    merged = df.merge(pore_scores[["record_id", "pore_teacher_score"]], left_on="sequence", right_on="record_id")
    if len(merged) != len(df):
        raise ValueError("Pore-Forming public teacher scores do not cover all public sequences.")
    membrane = MembraneMechanismExpert(input_dim=matrix.shape[1])
    mem_losses = train_binary_regressor(
        membrane,
        matrix,
        merged["pore_teacher_score"].to_numpy(dtype="float32"),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_checkpoint(
        membrane,
        Path(args.checkpoint_dir) / "membrane_expert.pt",
        metadata={"input_dim": matrix.shape[1], "losses": mem_losses, "source": "Pore-Forming"},
    )
    save_json({"state": "M_R0", "status": "public_pretrain_complete"}, Path(args.checkpoint_dir) / "metadata.json")
    print(f"M_R0 checkpoints written to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()

