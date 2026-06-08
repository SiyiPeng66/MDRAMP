"""Model-state metadata registry for closed-loop fine-tuning."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def write_state(
    state_name: str,
    output_path: Path,
    *,
    parent_state: str | None = None,
    round_data: str | None = None,
    notes: str = "",
) -> dict:
    state = {
        "state_name": state_name,
        "parent_state": parent_state,
        "round_data": round_data,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "frozen_components": [
            "model_weights",
            "feature_scaler",
            "target_set",
            "teacher_score_cache",
            "gate_parameters",
            "data_availability_cutoff",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-state", default=None)
    parser.add_argument("--round-data", default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    state = write_state(
        args.state,
        Path(args.output),
        parent_state=args.parent_state,
        round_data=args.round_data,
        notes=args.notes,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

