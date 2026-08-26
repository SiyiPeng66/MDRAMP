"""Train the lightweight, state-specific quantitative MIC regression head."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.models.mic_regressor import MICRegressionHead, censored_huber_loss
from src.train.train_utils import require_file, save_checkpoint


def train_mic_head(
    model: MICRegressionHead,
    peptide_embedding: np.ndarray,
    physchem: np.ndarray,
    expert_means: np.ndarray,
    target: np.ndarray,
    censored: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    device: str = "cpu",
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pep = torch.tensor(peptide_embedding, dtype=torch.float32)
    pc = torch.tensor(physchem, dtype=torch.float32)
    means = torch.tensor(expert_means, dtype=torch.float32)
    yy = torch.tensor(target, dtype=torch.float32)
    cc = torch.tensor(censored, dtype=torch.bool)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(yy))
        batch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = model(pep[idx].to(device), pc[idx].to(device), means[idx].to(device))
            loss = censored_huber_loss(pred, yy[idx].to(device), cc[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic-ledger", required=True)
    parser.add_argument("--peptide-features", required=True)
    parser.add_argument("--peptide-embeddings", required=True)
    parser.add_argument("--peptide-embedding-metadata", required=True)
    parser.add_argument("--expert-means", required=True, help="CSV keyed by peptide_id with S_prior,S_mem,S_int.")
    parser.add_argument("--peptide-id-column", default="private_record_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()

    ledger = pd.read_csv(require_file(args.mic_ledger))
    required = {"peptide_id", "mic_log2", "mic_censored"}
    if not required.issubset(ledger.columns):
        raise ValueError(f"MIC ledger missing columns: {sorted(required - set(ledger.columns))}")
    features = pd.read_csv(require_file(args.peptide_features))
    store = EmbeddingStore.load(args.peptide_embeddings, args.peptide_embedding_metadata)
    means = pd.read_csv(require_file(args.expert_means))
    mean_cols = ["S_prior", "S_mem", "S_int"]
    if not set(mean_cols).issubset(means.columns):
        raise ValueError(f"Expert means missing columns: {mean_cols}")
    key = args.peptide_id_column
    joined = ledger.merge(features[[key] + feature_columns(features)], left_on="peptide_id", right_on=key)
    joined = joined.merge(means[["peptide_id"] + mean_cols], on="peptide_id", validate="one_to_one")
    ids = joined["peptide_id"].astype(str).tolist()
    model = MICRegressionHead(
        peptide_dim=store.global_embeddings.shape[1],
        physchem_dim=len(feature_columns(features)),
    )
    losses = train_mic_head(
        model,
        store.global_by_record_id(ids),
        joined[feature_columns(features)].to_numpy(dtype=np.float32),
        joined[mean_cols].to_numpy(dtype=np.float32),
        joined["mic_log2"].to_numpy(dtype=np.float32),
        joined["mic_censored"].astype(bool).to_numpy(),
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device,
    )
    save_checkpoint(model, Path(args.output), metadata={
        "peptide_dim": store.global_embeddings.shape[1],
        "physchem_dim": len(feature_columns(features)),
        "expert_dim": 3,
        "losses": losses,
        "target": "log2(MIC)",
        "censor_limit_log2": 8.0,
    })
    print(f"MIC head checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
