"""Shared training utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns


def require_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def load_peptide_matrix(
    feature_csv: Path,
    embedding_npz: Path,
    embedding_metadata: Path,
    *,
    id_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(require_file(feature_csv))
    store = EmbeddingStore.load(require_file(embedding_npz), require_file(embedding_metadata))
    embeddings = store.global_by_record_id([str(value) for value in df[id_column]])
    pcs = df[feature_columns(df)].to_numpy(dtype=np.float32)
    matrix = np.concatenate([embeddings.astype(np.float32), pcs], axis=1)
    return df, matrix


def save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def train_binary_regressor(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
) -> list[float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCELoss()
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    losses: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses

