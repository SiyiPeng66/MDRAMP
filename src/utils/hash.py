"""File hashing utilities for provenance tracking."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing(paths: list[str | Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        path = Path(path)
        if path.exists() and path.is_file():
            result[str(path)] = sha256_file(path)
    return result

