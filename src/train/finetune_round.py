"""Round fine-tuning from M_Rk to M_R(k+1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.features.physicochemical import add_physicochemical_features
from src.models.expert_modules import MembraneMechanismExpert
from src.train.train_utils import load_peptide_matrix, require_file, save_checkpoint, train_binary_regressor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-csv", default="outputs/processed/private_R0.csv")
    parser.add_argument("--features", default="outputs/processed/private_R0_features.csv")
    parser.add_argument("--embeddings", default="outputs/esm3_embeddings/private_R0_peptides.npz")
    parser.add_argument("--embedding-metadata", default="outputs/esm3_embeddings/private_R0_peptides_metadata.csv")
    parser.add_argument("--pore-scores", default="outputs/teacher_scores/pore_private_R0.csv")
    parser.add_argument("--from-checkpoint", default="outputs/checkpoints/M_R0/membrane_expert.pt")
    parser.add_argument("--output", default="outputs/checkpoints/M_R1/membrane_expert.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    add_physicochemical_features(Path(args.round_csv), Path(args.features), sequence_column="sequence")
    df, matrix = load_peptide_matrix(
        Path(args.features),
        Path(args.embeddings),
        Path(args.embedding_metadata),
        id_column="private_record_id",
    )
    ckpt = torch.load(require_file(args.from_checkpoint), map_location="cpu")
    model = MembraneMechanismExpert(input_dim=ckpt["metadata"]["input_dim"])
    model.load_state_dict(ckpt["state_dict"])

    pore_scores = pd.read_csv(require_file(args.pore_scores))
    merged = df.merge(pore_scores[["record_id", "pore_teacher_score"]], left_on="private_record_id", right_on="record_id")
    if len(merged) != len(df):
        raise ValueError("Pore-Forming private teacher scores do not cover all round sequences.")
    losses = train_binary_regressor(
        model,
        matrix,
        merged["pore_teacher_score"].to_numpy(dtype="float32"),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_checkpoint(model, Path(args.output), metadata={**ckpt["metadata"], "round_finetune_losses": losses})
    print(f"round fine-tuned membrane checkpoint -> {args.output}")


if __name__ == "__main__":
    main()

