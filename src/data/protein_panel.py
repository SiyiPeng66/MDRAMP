"""Deterministic construction and freezing of a multi-target protein panel."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_protein_sequence


def build_protein_panel(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    expected_working_corpus_size: int = 21_445,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {
        "target_id", "target_sequence", "target_pathogen", "functional_category",
        "essentiality_resistance_relevance", "annotation_quality", "redundancy_cluster",
    }
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
    if len(result) != expected_working_corpus_size:
        raise ValueError(
            f"Frozen working protein corpus must contain {expected_working_corpus_size} unique valid sequences; got {len(result)}"
        )
    result["annotation_quality"] = pd.to_numeric(result["annotation_quality"], errors="raise")
    result["essentiality_resistance_relevance"] = pd.to_numeric(
        result["essentiality_resistance_relevance"], errors="raise"
    )
    result = (
        result.sort_values(
            ["essentiality_resistance_relevance", "annotation_quality", "target_id"],
            ascending=[False, False, True],
        )
        .drop_duplicates("redundancy_cluster")
        .sort_values(["target_pathogen", "functional_category", "target_id"])
        .reset_index(drop=True)
    )
    result["panel_index"] = range(len(result))
    result["panel_hash"] = hashlib.sha256(
        "\n".join(result["target_id"].astype(str) + ":" + result["target_sequence"]).encode()
    ).hexdigest()
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
