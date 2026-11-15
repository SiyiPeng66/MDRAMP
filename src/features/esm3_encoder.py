"""Strict ESM3 embedding wrapper.

This module intentionally implements no substitute encoder. If ESM3 is not
installed or the configured checkpoint cannot be loaded, embedding generation
fails with an explicit setup error.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


_SDK_LOAD_LOCK = threading.RLock()


class ESM3SetupError(RuntimeError):
    """Raised when the required ESM3 dependency or checkpoint is unavailable."""


@dataclass(frozen=True)
class ESM3Embedding:
    """Embeddings for one protein or peptide sequence."""

    record_id: str
    sequence: str
    residue_embedding: np.ndarray
    global_embedding: np.ndarray


def _import_esm3() -> tuple[Any, Any, Any]:
    """Import ESM3 objects lazily so module imports remain lightweight."""

    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise ESM3SetupError(
            "ESM3 is required for formal embedding generation but is not "
            "available in this environment. Install it with: "
            "pip install esm@git+https://github.com/Biohub/esm.git@main"
        ) from exc
    return ESM3, ESMProtein, LogitsConfig


def resolve_device(configured_device: str) -> torch.device:
    """Resolve the configured device without silently changing model choice."""

    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ESM3SetupError(
            "configs/env.yaml requests device=cuda, but CUDA is not available. "
            "Set runtime.device to cpu only if you intentionally want CPU ESM3 inference."
        )
    return torch.device(configured_device)


class ESM3Encoder:
    """Load ESM3 and produce residue-level and global sequence embeddings."""

    def __init__(
        self,
        *,
        model_name: str = "esm3-sm-open-v1",
        device: str = "cuda",
        weights_path: str | Path | None = None,
    ) -> None:
        ESM3, ESMProtein, LogitsConfig = _import_esm3()
        self._ESMProtein = ESMProtein
        self._LogitsConfig = LogitsConfig
        self.device = resolve_device(device)
        self.model_name = model_name
        try:
            with _SDK_LOAD_LOCK:
                if weights_path is None:
                    self.model = ESM3.from_pretrained(model_name, device=self.device)
                else:
                    # The official 3.2 SDK resolves all tokenizer resources via data_root.
                    # Bind that resolver only during initialization to a complete pinned
                    # snapshot, then restore it. Sequence-only inference uses no lazy decoders.
                    weight = Path(weights_path).expanduser().absolute()
                    if weight.name != "esm3_sm_open_v1.pth" or not weight.is_file():
                        raise ValueError("esm_checkpoint must be the snapshot data/weights/esm3_sm_open_v1.pth path")
                    snapshot = weight.parent.parent.parent
                    if weight.parent.name != "weights" or weight.parent.parent.name != "data":
                        raise ValueError("ESM3 weights require their companion snapshot data directory")
                    import esm.pretrained as pretrained
                    import esm.utils.constants.esm3 as constants
                    previous_constants, previous_pretrained = constants.data_root, pretrained.data_root
                    def local_root(name):
                        if not name.startswith("esm3"):
                            raise ValueError("Pinned ESM3 snapshot cannot load a different model")
                        return snapshot
                    try:
                        constants.data_root = local_root
                        pretrained.data_root = local_root
                        self.model = ESM3.from_pretrained(model_name, device=self.device)
                    finally:
                        constants.data_root, pretrained.data_root = previous_constants, previous_pretrained
        except Exception as exc:  # pragma: no cover - depends on external weights
            raise ESM3SetupError(
                f"Failed to load required ESM3 checkpoint {model_name!r}. "
                "Check the ESM installation, model access, and local cache."
            ) from exc
        self.model.eval()

    @torch.no_grad()
    def encode_sequence(self, record_id: str, sequence: str) -> ESM3Embedding:
        """Encode one sequence with ESM3."""

        protein = self._ESMProtein(sequence=sequence)
        encoded = self.model.encode(protein)
        logits_config = self._LogitsConfig(
            sequence=True,
            return_embeddings=True,
        )
        output = self.model.logits(encoded, logits_config)
        residue = _tensor_to_numpy(output.embeddings)
        residue = _align_residue_embeddings(residue, expected_length=len(sequence))
        global_embedding = residue.mean(axis=0)
        return ESM3Embedding(
            record_id=record_id,
            sequence=sequence,
            residue_embedding=residue.astype(np.float32, copy=False),
            global_embedding=global_embedding.astype(np.float32, copy=False),
        )


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors or tensor-like outputs to a NumPy array."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _align_residue_embeddings(embeddings: np.ndarray, *, expected_length: int) -> np.ndarray:
    """Align model embeddings to the biological sequence length.

    ESM SDK outputs may include a batch dimension and may include special tokens
    depending on version. This function removes those dimensions deterministically
    and raises if the result cannot be aligned exactly.
    """

    array = embeddings
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 from ESM3, got {array.shape[0]}.")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D residue embeddings, got shape {array.shape}.")

    if array.shape[0] == expected_length:
        return array
    if array.shape[0] == expected_length + 2:
        return array[1:-1]
    if array.shape[0] == expected_length + 1:
        return array[1:]
    raise ValueError(
        "Could not align ESM3 embeddings to sequence length: "
        f"embedding rows={array.shape[0]}, sequence length={expected_length}."
    )


def save_embeddings_npz(
    embeddings: list[ESM3Embedding],
    *,
    output_npz: Path,
    metadata_csv: Path,
    metadata_json: Path,
    model_name: str,
) -> None:
    """Save variable-length residue embeddings and fixed global embeddings."""

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    metadata_rows = []
    global_rows = []

    for idx, item in enumerate(embeddings):
        key = f"emb_{idx:08d}"
        arrays[key] = item.residue_embedding
        global_rows.append(item.global_embedding)
        metadata_rows.append(
            {
                "record_id": item.record_id,
                "sequence": item.sequence,
                "length": len(item.sequence),
                "embedding_key": key,
                "global_index": idx,
            }
        )

    arrays["global_embeddings"] = (
        np.vstack(global_rows).astype(np.float32, copy=False)
        if global_rows
        else np.empty((0, 0), dtype=np.float32)
    )
    np.savez_compressed(output_npz, **arrays)

    import pandas as pd

    pd.DataFrame(metadata_rows).to_csv(metadata_csv, index=False)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": model_name,
                "encoder": "ESM3",
                "record_count": len(embeddings),
                "npz": str(output_npz),
                "metadata_csv": str(metadata_csv),
            },
            handle,
            indent=2,
        )

