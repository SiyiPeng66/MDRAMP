"""Apply independent HemoPI-2 risk after antibacterial scoring for R5."""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
import pandas as pd

from src.external.hemopi_wrapper import HemoPI2Teacher
from src.selection.pareto import pareto_front


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--antibacterial-scores", required=True)
    parser.add_argument("--hemopi-model", required=True)
    parser.add_argument("--hemopi-command", required=True, help="Quoted argv template with {model} and {sequence}")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--activity-column", default="R_t")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.antibacterial_scores)
    required = {args.sequence_column, args.activity_column}
    if not required.issubset(frame): raise ValueError(f"R5 input missing {sorted(required - set(frame))}")
    teacher = HemoPI2Teacher(args.hemopi_model, shlex.split(args.hemopi_command))
    frame["hemolysis_risk"] = frame[args.sequence_column].astype(str).map(teacher.score)
    selected = pareto_front(frame, activity=args.activity_column, risk="hemolysis_risk")
    selected = selected.sort_values([args.activity_column, "hemolysis_risk"], ascending=[False, True])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); selected.to_csv(args.output, index=False)


if __name__ == "__main__": main()
