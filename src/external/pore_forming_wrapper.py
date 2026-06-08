"""Wrapper for the required Pore-Forming teacher model."""

from __future__ import annotations

import argparse
import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_pore_module(model_root: Path) -> Any:
    predict_path = model_root / "predict.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"Pore-Forming predict.py not found: {predict_path}")
    spec = importlib.util.spec_from_file_location("pore_forming_predict", predict_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Pore-Forming module from {predict_path}")
    module = importlib.util.module_from_spec(spec)
    with pushd(model_root):
        spec.loader.exec_module(module)
    return module


class PoreFormingTeacher:
    """Required membrane-mechanism teacher wrapper."""

    def __init__(self, model_root: str | Path = "otherModels/Pore-Forming") -> None:
        self.model_root = Path(model_root)
        self.module = load_pore_module(self.model_root)

    def score(self, sequence: str) -> float:
        """Return a pore-forming probability when available, else hard label.

        The formal preferred path is `predict_proba`. If the upstream script has
        not yet been patched to expose probabilities, this returns its hard
        prediction as a float while preserving the provenance as teacher output.
        """

        sequence = str(sequence).strip().upper()
        if hasattr(self.module, "BERT_Extracting_seq_features") and hasattr(
            self.module, "clf_choice"
        ):
            embedding = self.module.BERT_Extracting_seq_features(sequence)
            clf = self.module.clf_choice
            if hasattr(clf, "predict_proba"):
                return float(clf.predict_proba(embedding)[0][1])
            return float(clf.predict(embedding)[0])
        if hasattr(self.module, "predict_amp"):
            return float(self.module.predict_amp(sequence))
        raise AttributeError("Pore-Forming module exposes neither predict_proba path nor predict_amp.")


def score_csv(
    input_csv: Path,
    output_csv: Path,
    *,
    model_root: Path,
    sequence_column: str = "sequence",
    id_column: str = "sequence",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    teacher = PoreFormingTeacher(model_root)
    rows = []
    for row in df.itertuples(index=False):
        sequence = getattr(row, sequence_column)
        record_id = getattr(row, id_column)
        rows.append(
            {
                "record_id": record_id,
                "sequence": sequence,
                "pore_teacher_score": teacher.score(sequence),
                "teacher_model": "Pore-Forming",
            }
        )
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default="otherModels/Pore-Forming")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--id-column", default="sequence")
    args = parser.parse_args()
    df = score_csv(
        Path(args.input),
        Path(args.output),
        model_root=Path(args.model_root),
        sequence_column=args.sequence_column,
        id_column=args.id_column,
    )
    print(f"Pore-Forming scores: {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()

