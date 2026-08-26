"""Operational unlabelled background construction for the AMP Prior Expert."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_peptide_sequence


DEFAULT_EXCLUSION_TERMS = (
    "antimicrobial", "amp", "defensin", "bacteriocin", "toxin", "venom"
)


def build_operational_background(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    sequence_column: str = "sequence",
    description_columns: tuple[str, ...] = ("description", "annotation", "keywords"),
    target_lengths: pd.Series | None = None,
    exclusion_terms: tuple[str, ...] = DEFAULT_EXCLUSION_TERMS,
) -> pd.DataFrame:
    """Filter a protein snapshot into length-matched operational background.

    The output is explicitly marked ``label_type=unlabelled``; it is never a
    confirmed-negative set. Close-homology removal requires a project-specific
    homology index and is therefore exposed as a later filter rather than
    silently approximated here.
    """

    df = pd.read_csv(input_csv)
    if sequence_column not in df.columns:
        raise ValueError(f"Background input missing sequence column: {sequence_column}")
    present_description = [c for c in description_columns if c in df.columns]
    text = (
        df[present_description].fillna("").astype(str).agg(" ".join, axis=1)
        if present_description
        else pd.Series("", index=df.index)
    )
    term_re = re.compile("|".join(re.escape(t) for t in exclusion_terms), re.I)
    clean = df[sequence_column].map(clean_peptide_sequence)
    result = df.assign(
        sequence=clean.map(lambda x: x.sequence),
        is_valid=clean.map(lambda x: x.is_valid),
        clean_reason=clean.map(lambda x: x.reason),
        label_type="unlabelled",
    )
    excluded = text.str.contains(term_re, na=False)
    result = result[result["is_valid"] & ~excluded].copy()
    if target_lengths is not None and len(target_lengths):
        lengths = set(target_lengths.astype(int).tolist())
        result = result[result["sequence"].str.len().isin(lengths)]
    result = result.drop_duplicates("sequence").reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
