"""Run a dependency-light smoke test for the replication foundation layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loop.prequential_guard import assert_no_current_round_labels, assert_round_cutoff
from src.loop.state_manifest import StateManifest, load_manifest, write_manifest
from src.models.mic_regressor import MICRegressionHead, activity_priority, censored_huber_loss


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "foundation_state_manifest.json"
    write_manifest(
        StateManifest(
            state_name="M_R1",
            parent_state="M_R0",
            visible_round="R1",
            data_cutoff="fixture-before-R1-release",
            input_hashes={"fixture": "synthetic"},
        ),
        manifest_path,
    )
    manifest = load_manifest(manifest_path)
    manifest.assert_paper_policy()

    visible = pd.DataFrame(
        {
            "round": ["R0", "R1"],
            "mic_log2": [6.0, None],
        }
    )
    assert_round_cutoff(visible, visible_round=manifest.visible_round)
    assert_no_current_round_labels(
        visible,
        current_round="R1",
        label_columns=["mic_log2"],
    )

    head = MICRegressionHead(peptide_dim=8, physchem_dim=4)
    peptide = torch.zeros(2, 8)
    physchem = torch.zeros(2, 4)
    expert_means = torch.zeros(2, 3)
    prediction = head(peptide, physchem, expert_means)
    target = torch.tensor([6.0, 8.0])
    censored = torch.tensor([False, True])
    loss = censored_huber_loss(prediction, target, censored)
    priority = activity_priority(prediction)

    result = {
        "state_name": manifest.state_name,
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "activity_priority_range": [
            float(priority.min().detach()),
            float(priority.max().detach()),
        ],
        "status": "foundation_dry_run_passed",
    }
    (output_dir / "foundation_dry_run.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/dry_run/foundation")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
