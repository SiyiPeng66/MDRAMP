"""Serializable, auditable metadata for one frozen MDRAMP model state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StateManifest:
    """The information required to replay a model state without leakage."""

    state_name: str
    parent_state: str | None
    visible_round: str
    data_cutoff: str
    frozen_components: tuple[str, ...] = (
        "esm3_encoder",
        "feature_scaler",
        "protein_panel",
        "expert_ensembles",
        "teacher_score_cache",
        "gate_parameters",
        "selection_policy",
    )
    input_hashes: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frozen_components"] = list(self.frozen_components)
        return payload

    def assert_paper_policy(self) -> None:
        """Validate the non-negotiable frozen components for R0-R3."""

        required = {
            "esm3_encoder", "feature_scaler", "protein_panel",
            "expert_ensembles", "gate_parameters", "selection_policy",
        }
        missing = required - set(self.frozen_components)
        if missing:
            raise ValueError(f"State manifest is not paper-complete; missing frozen components: {sorted(missing)}")
        if self.config.get("update_component") == "expert_weights":
            raise ValueError("Paper R0-R3 state updates must not update Expert weights.")


def write_manifest(manifest: StateManifest, output_path: str | Path) -> dict[str, Any]:
    """Write a state manifest and return its JSON-compatible payload."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def load_manifest(path: str | Path) -> StateManifest:
    """Load and validate a manifest written by :func:`write_manifest`."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"state_name", "visible_round", "data_cutoff", "frozen_components"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"State manifest missing required fields: {missing}")
    return StateManifest(
        state_name=str(payload["state_name"]),
        parent_state=payload.get("parent_state"),
        visible_round=str(payload["visible_round"]),
        data_cutoff=str(payload["data_cutoff"]),
        frozen_components=tuple(payload["frozen_components"]),
        input_hashes=dict(payload.get("input_hashes", {})),
        config=dict(payload.get("config", {})),
        created_at_utc=str(payload.get("created_at_utc", "")),
    )
