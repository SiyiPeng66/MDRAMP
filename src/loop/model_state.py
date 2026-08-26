"""Model-state metadata registry for closed-loop fine-tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.loop.state_manifest import StateManifest, write_manifest


def write_state(
    state_name: str,
    output_path: Path,
    *,
    parent_state: str | None = None,
    round_data: str | None = None,
    notes: str = "",
    visible_round: str | None = None,
    data_cutoff: str | None = None,
    input_hashes: dict[str, str] | None = None,
) -> dict:
    visible_round = visible_round or state_name.replace("M_", "")
    data_cutoff = data_cutoff or datetime.now(timezone.utc).isoformat()
    manifest = StateManifest(
        state_name=state_name,
        parent_state=parent_state,
        visible_round=visible_round,
        data_cutoff=data_cutoff,
        input_hashes=input_hashes or {},
        config={"round_data": round_data, "notes": notes},
    )
    return write_manifest(manifest, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-state", default=None)
    parser.add_argument("--round-data", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--visible-round", default=None)
    parser.add_argument("--data-cutoff", default=None)
    args = parser.parse_args()
    state = write_state(
        args.state,
        Path(args.output),
        parent_state=args.parent_state,
        round_data=args.round_data,
        notes=args.notes,
        visible_round=args.visible_round,
        data_cutoff=args.data_cutoff,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
