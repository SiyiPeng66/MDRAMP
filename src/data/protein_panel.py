"""Deterministic construction and freezing of a multi-target protein panel."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_protein_sequence


def build_protein_panel(input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"target_id", "target_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Protein panel input missing columns: {sorted(missing)}")
    cleaned = df["target_sequence"].map(clean_protein_sequence)
    result = df.assign(
        target_sequence=cleaned.map(lambda x: x.sequence),
        is_valid=cleaned.map(lambda x: x.is_valid),
        clean_reason=cleaned.map(lambda x: x.reason),
    )
    result = result[result["is_valid"]].drop_duplicates("target_sequence").reset_index(drop=True)
    result["panel_index"] = range(len(result))
    result["panel_hash"] = hashlib.sha256(
        "\n".join(result["target_id"].astype(str) + ":" + result["target_sequence"]).encode()
    ).hexdigest()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
