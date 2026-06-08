"""Predict pair-level target affinity and aggregate S_aff with Noisy-OR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.external.tpeppro_wrapper import aggregate_pair_scores
from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.models.expert_modules import PairAffinityExpert
from src.train.train_utils import require_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--peptide-features", required=True)
    parser.add_argument("--peptide-embeddings", required=True)
    parser.add_argument("--peptide-embedding-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True)
    parser.add_argument("--target-embedding-metadata", required=True)
    parser.add_argument("--peptide-id-column", default="private_record_id")
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--pair-output", default="outputs/teacher_scores/model_pair_affinity.csv")
    parser.add_argument("--aggregate-output", default="outputs/teacher_scores/model_aggregated_affinity.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(require_file(args.checkpoint), map_location="cpu")
    metadata = ckpt["metadata"]
    model = PairAffinityExpert(
        metadata["peptide_dim"],
        metadata["target_dim"],
        metadata["physchem_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    peptides = pd.read_csv(require_file(args.peptide_features))
    peptide_store = EmbeddingStore.load(args.peptide_embeddings, args.peptide_embedding_metadata)
    target_store = EmbeddingStore.load(args.target_embeddings, args.target_embedding_metadata)
    target_meta = target_store.metadata
    if args.target_limit is not None:
        target_meta = target_meta.head(args.target_limit)
    pc_cols = feature_columns(peptides)
    rows = []
    with torch.no_grad():
        for peptide in peptides.itertuples(index=False):
            peptide_id = str(getattr(peptide, args.peptide_id_column))
            pep_emb = peptide_store.global_by_record_id([peptide_id])[0]
            pc = np.asarray([getattr(peptide, col) for col in pc_cols], dtype=np.float32)
            for target in target_meta.itertuples(index=False):
                target_id = str(target.record_id)
                tar_emb = target_store.global_by_record_id([target_id])[0]
                score = model(
                    torch.tensor(pep_emb[None, :], dtype=torch.float32, device=args.device),
                    torch.tensor(tar_emb[None, :], dtype=torch.float32, device=args.device),
                    torch.tensor(pc[None, :], dtype=torch.float32, device=args.device),
                )
                rows.append(
                    {
                        "peptide_id": peptide_id,
                        "target_id": target_id,
                        "s_aff_pair": float(score.detach().cpu().item()),
                    }
                )
    pair_df = pd.DataFrame(rows)
    Path(args.pair_output).parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(args.pair_output, index=False)
    aggregate_pair_scores(Path(args.pair_output), Path(args.aggregate_output))
    print(f"pair scores -> {args.pair_output}")
    print(f"aggregated S_aff -> {args.aggregate_output}")


if __name__ == "__main__":
    main()

