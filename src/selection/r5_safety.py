"""Apply independent HemoPI-2 risk after antibacterial scoring for R5."""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
import pandas as pd

from src.external.hemopi_wrapper import HemoPI2Teacher



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antibacterial-scores", required=True)
    parser.add_argument("--hemopi-model", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--hemopi-official-python", help="Isolated Python for official RF+MERCI batch runtime")
    modes.add_argument("--hemopi-command", help="Quoted argv template with {model} and {sequence}")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--activity-column", default="R_t")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.antibacterial_scores)
    required = {args.sequence_column, args.activity_column}
    if not required.issubset(frame): raise ValueError(f"R5 input missing {sorted(required - set(frame))}")
    if args.hemopi_official_python:
        from src.external.hemopi_official import score_batch
        frame["hemolysis_risk"] = score_batch(frame[args.sequence_column].astype(str).tolist(),args.hemopi_model,args.hemopi_official_python)
    else:
        teacher = HemoPI2Teacher(args.hemopi_model, shlex.split(args.hemopi_command))
        frame["hemolysis_risk"] = frame[args.sequence_column].astype(str).map(teacher.score)
    selected = frame.copy()
    selected = selected.sort_values(["hemolysis_risk", args.activity_column], ascending=[True, False], kind="stable")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); selected.to_csv(args.output, index=False)


if __name__ == "__main__": main()
