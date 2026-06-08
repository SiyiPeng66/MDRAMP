"""Peptide physicochemical descriptors used by the expert models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPATHY_KD = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWYV")
AROMATIC = set("FWY")
POLAR = set("STNQCY")
BOMAN_SOLUBILITY = {
    "A": 0.17,
    "C": -0.24,
    "D": 1.23,
    "E": 2.02,
    "F": -1.13,
    "G": 0.01,
    "H": 0.96,
    "I": -0.31,
    "K": 0.99,
    "L": -0.56,
    "M": -0.23,
    "N": 0.42,
    "P": 0.45,
    "Q": 0.58,
    "R": 0.81,
    "S": 0.13,
    "T": 0.14,
    "V": 0.07,
    "W": -1.85,
    "Y": -0.94,
}


def fraction(sequence: str, alphabet: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(1 for aa in sequence if aa in alphabet) / len(sequence)


def net_charge_ph74(sequence: str) -> float:
    """Approximate net charge at pH 7.4."""

    return (
        sequence.count("K")
        + sequence.count("R")
        + 0.1 * sequence.count("H")
        - sequence.count("D")
        - sequence.count("E")
    )


def approximate_hydrophobic_moment(sequence: str) -> float:
    """Approximate manuscript-style hydrophobic moment using i and i+3 gaps."""

    if len(sequence) <= 3:
        return 0.0
    diffs = [
        abs(HYDROPATHY_KD[sequence[i]] - HYDROPATHY_KD[sequence[i + 3]])
        for i in range(len(sequence) - 3)
    ]
    return float(np.mean(diffs))


def compute_descriptors(sequence: str) -> dict[str, float]:
    sequence = str(sequence).upper()
    length = len(sequence)
    hydrophobicities = [HYDROPATHY_KD[aa] for aa in sequence]
    boman_values = [BOMAN_SOLUBILITY[aa] for aa in sequence]
    row: dict[str, float] = {
        "pc_length": float(length),
        "pc_net_charge_ph74": float(net_charge_ph74(sequence)),
        "pc_mean_hydrophobicity_kd": float(np.mean(hydrophobicities)) if length else 0.0,
        "pc_hydrophobic_fraction": fraction(sequence, HYDROPHOBIC),
        "pc_positive_fraction": fraction(sequence, POSITIVE),
        "pc_negative_fraction": fraction(sequence, NEGATIVE),
        "pc_aromatic_fraction": fraction(sequence, AROMATIC),
        "pc_polar_fraction": fraction(sequence, POLAR),
        "pc_cysteine_fraction": fraction(sequence, {"C"}),
        "pc_proline_fraction": fraction(sequence, {"P"}),
        "pc_boman_index": float(np.mean(boman_values)) if length else 0.0,
        "pc_approx_hydrophobic_moment": approximate_hydrophobic_moment(sequence),
    }
    for aa in AA_ORDER:
        row[f"pc_aa_{aa}"] = sequence.count(aa) / length if length else 0.0
    return row


def add_physicochemical_features(
    input_csv: Path,
    output_csv: Path,
    *,
    sequence_column: str = "sequence",
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    features = pd.DataFrame([compute_descriptors(seq) for seq in df[sequence_column]])
    result = pd.concat([df.reset_index(drop=True), features], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("pc_")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-column", default="sequence")
    args = parser.parse_args()
    df = add_physicochemical_features(
        Path(args.input), Path(args.output), sequence_column=args.sequence_column
    )
    print(f"physicochemical features: {df.shape} -> {args.output}")


if __name__ == "__main__":
    main()

