"""Small FASTA helpers used by the data parsers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FastaRecord:
    """A parsed FASTA record."""

    record_id: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read FASTA records from a text file.

    Empty records, such as summary headers without a sequence, are skipped.
    """

    records: list[FastaRecord] = []
    current_header: str | None = None
    sequence_lines: list[str] = []
    path = Path(path)

    def flush() -> None:
        nonlocal current_header, sequence_lines
        if current_header is None:
            return
        sequence = "".join(sequence_lines).strip()
        if sequence:
            parts = current_header.strip().split(maxsplit=1)
            record_id = parts[0] if parts else ""
            description = parts[1] if len(parts) > 1 else ""
            records.append(FastaRecord(record_id, description, sequence))
        current_header = None
        sequence_lines = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line[1:].strip()
            else:
                sequence_lines.append(line)
        flush()

    return records


def write_fasta(records: Iterable[FastaRecord], path: str | Path) -> None:
    """Write FASTA records."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            description = f" {record.description}" if record.description else ""
            handle.write(f">{record.record_id}{description}\n")
            sequence = record.sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

