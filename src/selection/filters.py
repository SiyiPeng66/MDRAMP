"""Common candidate guardrails from the prospective selection protocol."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.data.clean_sequences import clean_peptide_sequence
from src.features.physicochemical import HYDROPHOBIC, compute_descriptors


def passes_guardrails(sequence: str, *, reference_sequences: Iterable[str] = ()) -> tuple[bool, str]:
    clean = clean_peptide_sequence(sequence)
    if not clean.is_valid:
        return False, clean.reason
    seq = clean.sequence
    desc = compute_descriptors(seq)
    if not 2.0 <= desc["pc_net_charge_ph74"] <= 10.0:
        return False, "net_charge_out_of_range"
    if not 0.30 <= desc["pc_hydrophobic_fraction"] <= 0.60:
        return False, "hydrophobic_fraction_out_of_range"
    if max(seq.count(aa) for aa in set(seq)) / len(seq) > 0.35:
        return False, "single_residue_fraction_too_high"
    if any(aa * 4 in seq for aa in set(seq)):
        return False, "homopolymer_run"
    run = 0
    for aa in seq:
        run = run + 1 if aa in HYDROPHOBIC else 0
        if run > 5:
            return False, "hydrophobic_run"
    if any(identity(seq, other) >= 0.90 for other in reference_sequences):
        return False, "homology_or_previous_testing"
    return True, "valid"


def identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


def enforce_pairwise_cosine(sequences: list[str], embeddings: np.ndarray, minimum: float = 0.10) -> list[int]:
    selected: list[int] = []
    vectors = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    normalized = vectors / norms
    for idx in range(len(sequences)):
        if all(1.0 - float(normalized[idx] @ normalized[j]) >= minimum for j in selected):
            selected.append(idx)
    return selected
