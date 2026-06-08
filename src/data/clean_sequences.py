"""Sequence cleaning rules shared by public, private, and target parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class CleanResult:
    sequence: str
    is_valid: bool
    reason: str


def clean_peptide_sequence(
    sequence: str,
    *,
    min_length: int = 10,
    max_length: int = 50,
    allowed_aas: set[str] | None = None,
) -> CleanResult:
    """Clean and validate a peptide sequence."""

    allowed_aas = allowed_aas or DEFAULT_ALLOWED_AAS
    cleaned = re.sub(r"[^A-Za-z]", "", str(sequence)).upper()
    if not cleaned:
        return CleanResult(cleaned, False, "empty")
    invalid = sorted(set(cleaned) - allowed_aas)
    if invalid:
        return CleanResult(cleaned, False, "non_standard_residue:" + "".join(invalid))
    if len(cleaned) < min_length:
        return CleanResult(cleaned, False, "too_short")
    if len(cleaned) > max_length:
        return CleanResult(cleaned, False, "too_long")
    return CleanResult(cleaned, True, "valid")


def clean_protein_sequence(sequence: str) -> CleanResult:
    """Clean target protein sequences.

    Target proteins are not restricted to peptide length 10-50, but are still
    restricted to the 20 standard amino acids for the formal ESM3 workflow.
    """

    cleaned = re.sub(r"[^A-Za-z]", "", str(sequence)).upper()
    if not cleaned:
        return CleanResult(cleaned, False, "empty")
    invalid = sorted(set(cleaned) - DEFAULT_ALLOWED_AAS)
    if invalid:
        return CleanResult(cleaned, False, "non_standard_residue:" + "".join(invalid))
    return CleanResult(cleaned, True, "valid")

