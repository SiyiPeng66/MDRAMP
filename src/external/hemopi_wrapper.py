"""Explicit adapter for the external HemoPI-2 risk model."""

from __future__ import annotations

from pathlib import Path
import subprocess
from collections.abc import Sequence


class HemoPIUnavailable(RuntimeError):
    pass


class HemoPI2Teacher:
    """Pinned command adapter for the external HemoPI-2 hybrid model.

    The command must print exactly one normalized score. Placeholders
    ``{model}`` and ``{sequence}`` are expanded without invoking a shell.
    """

    def __init__(self, model_path: str | Path, command: Sequence[str]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise HemoPIUnavailable(
                f"HemoPI-2 model is required for R5 but was not found: {self.model_path}"
            )
        if not command:
            raise HemoPIUnavailable("A pinned HemoPI-2 scoring command is required")
        self.command = tuple(command)

    def score(self, sequence: str) -> float:
        command = [part.format(model=str(self.model_path), sequence=sequence) for part in self.command]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            value = float(completed.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise HemoPIUnavailable(f"HemoPI-2 scoring failed for sequence {sequence!r}: {exc}") from exc
        if not 0.0 <= value <= 1.0:
            raise HemoPIUnavailable(f"HemoPI-2 returned a score outside [0, 1]: {value}")
        return value
