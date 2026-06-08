"""Checkpoint metadata helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.utils.hash import hash_existing


def checkpoint_metadata(
    *,
    state_name: str,
    model_class: str,
    training_config: dict[str, Any],
    input_files: list[str | Path],
    parent_state: str | None = None,
    round_data: str | None = None,
) -> dict[str, Any]:
    return {
        "state_name": state_name,
        "parent_state": parent_state,
        "round_data": round_data,
        "model_class": model_class,
        "training_config": training_config,
        "input_hashes": hash_existing(input_files),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    with (path.with_suffix(path.suffix + ".metadata.json")).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

