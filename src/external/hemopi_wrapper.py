"""Explicit adapter for the external HemoPI-2 risk model."""

from __future__ import annotations

from pathlib import Path


class HemoPIUnavailable(RuntimeError):
    pass


class HemoPI2Teacher:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise HemoPIUnavailable(
                f"HemoPI-2 model is required for R5 but was not found: {self.model_path}"
            )

    def score(self, sequence: str) -> float:
        raise HemoPIUnavailable(
            "HemoPI-2 adapter is present but its external runtime is not installed. "
            "Install the pinned HemoPI environment before running R5."
        )
