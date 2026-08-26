"""State-specific feature scaling with explicit fit provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class StateFeatureScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, columns: list[str]):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.where(np.asarray(scale) == 0, 1.0, scale).astype(np.float32)
        self.columns = list(columns)

    @classmethod
    def fit(cls, values: np.ndarray, columns: list[str]) -> "StateFeatureScaler":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(columns):
            raise ValueError("values must be 2D and match the feature columns")
        return cls(values.mean(axis=0), values.std(axis=0), columns)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != len(self.columns):
            raise ValueError("feature dimension does not match fitted scaler")
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict:
        return {"columns": self.columns, "mean": self.mean.tolist(), "scale": self.scale.tolist()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StateFeatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(payload["mean"]), np.asarray(payload["scale"]), payload["columns"])
