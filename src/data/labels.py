"""Experimental label rules for future supervised private rounds."""

from __future__ import annotations


def calculate_inhibition(
    od_treated: float,
    od_blank: float,
    od_control: float,
) -> float:
    """Calculate fixed-endpoint growth inhibition percentage."""

    denominator = od_control - od_blank
    if denominator == 0:
        raise ValueError("OD control and blank produce a zero denominator.")
    return (1.0 - ((od_treated - od_blank) / denominator)) * 100.0


def binary_mdr_label(label_4class: str) -> int:
    """Map manuscript four-level labels to binary MDR-active labels."""

    if label_4class in {"resistant-potent", "resistant-active"}:
        return 1
    if label_4class in {"reference-positive", "inactive"}:
        return 0
    raise ValueError(f"Unknown four-level label: {label_4class}")

