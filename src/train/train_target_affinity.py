"""Train Target Affinity Expert from TPepPro pair-level teacher scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import ProjectedPairAffinityExpert
from src.train.train_utils import require_file, save_checkpoint


def build_pair_arrays(
    pair_scores: pd.DataFrame,
    peptide_features: pd.DataFrame,
    peptide_store: EmbeddingStore,
    target_store: EmbeddingStore,
    *,
    peptide_id_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pc_cols = feature_columns(peptide_features)
    peptide_lookup = peptide_features.set_index(peptide_id_column)
    peptide_embeddings = []
    target_embeddings = []
    physchem = []
    labels = []
    for row in pair_scores.itertuples(index=False):
        peptide_id = str(row.peptide_id)
        target_id = str(row.target_id)
        peptide_embeddings.append(peptide_store.global_by_record_id([peptide_id])[0])
        target_embeddings.append(target_store.global_by_record_id([target_id])[0])
        physchem.append(peptide_lookup.loc[peptide_id, pc_cols].to_numpy(dtype=np.float32))
        labels.append(float(row.s_aff_pair))
    return (
        np.asarray(peptide_embeddings, dtype=np.float32),
        np.asarray(target_embeddings, dtype=np.float32),
        np.asarray(physchem, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def train_pair_model(
    model: ExpertEnsemble,
    peptide_x: np.ndarray,
    target_x: np.ndarray,
    physchem_x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
) -> list[list[float]]:
    loss_fn = torch.nn.BCELoss()
    pep = torch.tensor(peptide_x, dtype=torch.float32)
    tar = torch.tensor(target_x, dtype=torch.float32)
    pc = torch.tensor(physchem_x, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.float32)
    histories = []
    for seed, member in zip(model.seeds, model.members):
        member.to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(epochs):
            batch_losses = []
            for idx in torch.randperm(len(yy), generator=generator).split(batch_size):
                pred = member(pep[idx].to(device), tar[idx].to(device), pc[idx].to(device))
                loss = loss_fn(pred, yy[idx].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(batch_losses)))
        histories.append(losses)
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-scores", required=True)
    parser.add_argument("--peptide-features", required=True)
    parser.add_argument("--peptide-embeddings", required=True)
    parser.add_argument("--peptide-embedding-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True)
    parser.add_argument("--target-embedding-metadata", required=True)
    parser.add_argument("--peptide-id-column", default="sequence")
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--output", default="outputs/checkpoints/M_R0/target_affinity_expert.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    pair_scores = pd.read_csv(require_file(args.pair_scores))
    if "label_type" not in pair_scores:
        raise ValueError("Paper training pairs require label_type=positive/background")
    label_type = pair_scores["label_type"].astype(str)
    unexpected = sorted(set(label_type) - {"positive", "background"})
    if unexpected:
        raise ValueError(f"Unexpected pair label types: {unexpected}")
    counts = label_type.value_counts()
    if counts.get("positive", 0) != counts.get("background", 0):
        raise ValueError("Paper interaction training requires 1:1 positive/background pairs")
    peptide_features = pd.read_csv(require_file(args.peptide_features))
    pc_columns = feature_columns(peptide_features)
    scaler = StateFeatureScaler.load(require_file(args.scaler))
    if scaler.columns != pc_columns:
        raise ValueError("Interaction features do not match the frozen scaler")
    peptide_features.loc[:, pc_columns] = scaler.transform(
        peptide_features[pc_columns].to_numpy(dtype=np.float32)
    )
    peptide_store = EmbeddingStore.load(args.peptide_embeddings, args.peptide_embedding_metadata)
    target_store = EmbeddingStore.load(args.target_embeddings, args.target_embedding_metadata)
    peptide_x, target_x, physchem_x, y = build_pair_arrays(
        pair_scores,
        peptide_features,
        peptide_store,
        target_store,
        peptide_id_column=args.peptide_id_column,
    )
    peptide_dim = peptide_store.global_embeddings.shape[1]
    target_dim = target_store.global_embeddings.shape[1]
    physchem_dim = len(feature_columns(peptide_features))
    seeds = (11, 23, 37, 51, 73)
    model = ExpertEnsemble(
        lambda: ProjectedPairAffinityExpert(peptide_dim, target_dim, physchem_dim),
        seeds=seeds,
    )
    losses = train_pair_model(
        model,
        peptide_x,
        target_x,
        physchem_x,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    save_checkpoint(
        model,
        Path(args.output),
        metadata={
            "peptide_dim": peptide_dim,
            "target_dim": target_dim,
            "physchem_dim": physchem_dim,
            "losses": losses,
            "source": "TPepPro",
            "architecture": "ProjectedPairAffinityExpert",
            "projection_dim": 256,
            "bilinear_rank": 64,
            "seeds": list(seeds),
            "pair_sampling": "1:1 positive/operational_background",
            "scaler": str(Path(args.scaler)),
        },
    )
    print(f"target affinity checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
