"""Predict five-member pair evidence and aggregate interaction coverage."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.external.tpeppro_wrapper import aggregate_pair_frame
from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import ProjectedPairAffinityExpert
from src.train.train_utils import require_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--peptide-features", required=True)
    parser.add_argument("--peptide-embeddings", required=True)
    parser.add_argument("--peptide-embedding-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True)
    parser.add_argument("--target-embedding-metadata", required=True)
    parser.add_argument("--peptide-id-column", default="private_record_id")
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--pair-output", default="outputs/teacher_scores/model_pair_affinity.csv")
    parser.add_argument("--aggregate-output", default="outputs/teacher_scores/model_aggregated_affinity.csv")
    parser.add_argument("--aggregation-coefficient", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt = torch.load(require_file(args.checkpoint), map_location="cpu")
    meta = ckpt["metadata"]
    if meta.get("architecture") != "ProjectedPairAffinityExpert":
        raise ValueError("Paper mode requires a ProjectedPairAffinityExpert ensemble checkpoint")
    model = ExpertEnsemble(
        lambda: ProjectedPairAffinityExpert(meta["peptide_dim"], meta["target_dim"], meta["physchem_dim"]),
        seeds=meta["seeds"],
    )
    model.load_state_dict(ckpt["state_dict"]); model.to(args.device); model.eval()
    peptides = pd.read_csv(require_file(args.peptide_features))
    peptide_store = EmbeddingStore.load(args.peptide_embeddings, args.peptide_embedding_metadata)
    target_store = EmbeddingStore.load(args.target_embeddings, args.target_embedding_metadata)
    target_meta = target_store.metadata.head(args.target_limit) if args.target_limit else target_store.metadata
    pc_cols = feature_columns(peptides)
    scaler = StateFeatureScaler.load(args.scaler)
    if scaler.columns != pc_cols: raise ValueError("Scoring descriptors do not match the frozen scaler")
    scaled_pc = scaler.transform(peptides[pc_cols].to_numpy(dtype=np.float32))
    rows = []
    with torch.no_grad():
        for row_index, peptide in enumerate(peptides.itertuples(index=False)):
            peptide_id = str(getattr(peptide, args.peptide_id_column))
            pep = torch.as_tensor(peptide_store.global_by_record_id([peptide_id]), device=args.device)
            pc = torch.as_tensor(scaled_pc[row_index:row_index + 1], device=args.device)
            for target in target_meta.itertuples(index=False):
                target_id = str(target.record_id)
                tar = torch.as_tensor(target_store.global_by_record_id([target_id]), device=args.device)
                member_outputs = [member.forward_with_attention(pep, tar, pc) for member in model.members]
                scores = np.asarray([item["probability"].item() for item in member_outputs])
                attention = np.asarray([item["attention_logit"].item() for item in member_outputs])
                record = {"peptide_id": peptide_id, "target_id": target_id, "s_aff_pair": float(scores.mean())}
                record.update({f"member_{i}": float(score) for i, score in enumerate(scores)})
                record.update({f"attention_{i}": float(score) for i, score in enumerate(attention)})
                rows.append(record)
    pair_df = pd.DataFrame(rows)
    aggregate_members = []
    for i in range(len(model.members)):
        member_frame = pair_df[["peptide_id", "target_id", f"member_{i}", f"attention_{i}"]].rename(columns={f"member_{i}": "s_aff_pair", f"attention_{i}": "attention_logit"})
        aggregate = aggregate_pair_frame(member_frame, attention_noisy_or_weight=args.aggregation_coefficient)
        aggregate_members.append(aggregate.set_index("peptide_id")["S_aff"].rename(f"member_{i}"))
    result = pd.concat(aggregate_members, axis=1)
    result["S_int_mean"] = result.mean(axis=1)
    result["S_int_std"] = result[[f"member_{i}" for i in range(len(model.members))]].std(axis=1, ddof=0)
    best = pair_df.sort_values("s_aff_pair", ascending=False).groupby("peptide_id").first()
    result["top_target_id"] = best["target_id"]
    result["top_target_score"] = best["s_aff_pair"]
    result = result.reset_index()
    Path(args.pair_output).parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(args.pair_output, index=False)
    result.to_csv(args.aggregate_output, index=False)


if __name__ == "__main__": main()
