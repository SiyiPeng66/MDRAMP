"""Load cached ESM3 embeddings produced by build_esm3_embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EmbeddingStore:
    metadata: pd.DataFrame
    global_embeddings: np.ndarray
    npz: np.lib.npyio.NpzFile

    @classmethod
    def load(cls, npz_path: str | Path, metadata_csv: str | Path) -> "EmbeddingStore":
        npz = np.load(npz_path)
        metadata = pd.read_csv(metadata_csv)
        global_embeddings = np.asarray(npz["global_embeddings"], dtype=np.float32)
        return cls(metadata=metadata, global_embeddings=global_embeddings, npz=npz)

    def global_by_record_id(self, record_ids: list[str]) -> np.ndarray:
        lookup = {
            str(row.record_id): int(row.global_index)
            for row in self.metadata.itertuples(index=False)
        }
        indices = [lookup[str(record_id)] for record_id in record_ids]
        return self.global_embeddings[indices]

    def residue_by_record_id(self, record_id: str) -> np.ndarray:
        row = self.metadata[self.metadata["record_id"].astype(str) == str(record_id)]
        if row.empty:
            raise KeyError(f"record_id not found in embedding metadata: {record_id}")
        key = str(row.iloc[0]["embedding_key"])
        return np.asarray(self.npz[key], dtype=np.float32)

