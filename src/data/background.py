"""Operational unlabelled background construction for the AMP Prior Expert."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.clean_sequences import clean_peptide_sequence
from src.data.cluster_split import length_aware_identity


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
    amp_sequences: tuple[str, ...] = (),
    identity_threshold: float = 0.90,
    seed: int = 2024,
    exclusion_terms: tuple[str, ...] = DEFAULT_EXCLUSION_TERMS,
) -> pd.DataFrame:
    """Filter a protein snapshot into length-matched operational background.

    The output is explicitly marked ``label_type=unlabelled``; it is never a
    confirmed-negative set. Length matching preserves the complete positive
    length histogram, and exact/global close homologues are removed.
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
    if amp_sequences:
        amp_by_length: dict[int, list[str]] = {}
        for sequence in amp_sequences:
            amp_by_length.setdefault(len(sequence), []).append(sequence)
        def is_close(sequence: str) -> bool:
            low = int(len(sequence) * identity_threshold)
            high = int(len(sequence) / identity_threshold) + 1
            return any(
                length_aware_identity(sequence, amp) >= identity_threshold
                for length in range(low, high + 1)
                for amp in amp_by_length.get(length, ())
            )
        result = result[~result["sequence"].map(is_close)]
    if target_lengths is not None and len(target_lengths):
        required_counts = target_lengths.astype(int).value_counts().sort_index()
        result = result.assign(_length=result["sequence"].str.len())
        sampled = []
        for length, count in required_counts.items():
            available = result[result["_length"] == length]
            if len(available) < count:
                raise ValueError(f"Insufficient operational background at length {length}: {len(available)} < {count}")
            sampled.append(available.sample(n=int(count), random_state=seed + int(length)))
        result = pd.concat(sampled, ignore_index=True).drop(columns="_length")
    result = result.drop_duplicates("sequence").reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
