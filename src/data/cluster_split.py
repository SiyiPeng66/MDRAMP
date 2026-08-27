"""Deterministic 90% identity clustering and split utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterAssignment:
    sequence: str
    cluster_id: str


def length_aware_identity(seq_a: str, seq_b: str) -> float:
    """Needleman-Wunsch global identity, including insertion/deletion gaps."""

    if not seq_a or not seq_b:
        return 0.0
    # Each cell stores (matches, alignment_length); maximize identity numerator
    # first and use the shortest alignment to break ties.
    previous = [(0, j) for j in range(len(seq_b) + 1)]
    for i, aa in enumerate(seq_a, 1):
        current = [(0, i)]
        for j, bb in enumerate(seq_b, 1):
            candidates = (
                (previous[j - 1][0] + int(aa == bb), previous[j - 1][1] + 1),
                (previous[j][0], previous[j][1] + 1),
                (current[j - 1][0], current[j - 1][1] + 1),
            )
            current.append(max(candidates, key=lambda item: (item[0], -item[1])))
        previous = current
    matches, aligned = previous[-1]
    return matches / aligned


def greedy_cluster_sequences(
    sequences: list[str],
    *,
    identity_threshold: float = 0.90,
) -> list[ClusterAssignment]:
    """Cluster sequences greedily with a deterministic length-bucket index."""

    representatives: list[str] = []
    rep_by_length: dict[int, list[int]] = defaultdict(list)
    assignments: list[ClusterAssignment] = []

    for sequence in sorted(sequences, key=lambda item: (len(item), item)):
        seq_len = len(sequence)
        assigned_idx: int | None = None
        min_rep_len = max(1, int(seq_len * identity_threshold))
        max_rep_len = int(seq_len / identity_threshold) + 1
        for rep_len in range(min_rep_len, max_rep_len + 1):
            for rep_idx in rep_by_length.get(rep_len, []):
                if length_aware_identity(sequence, representatives[rep_idx]) >= identity_threshold:
                    assigned_idx = rep_idx
                    break
            if assigned_idx is not None:
                break

        if assigned_idx is None:
            assigned_idx = len(representatives)
            representatives.append(sequence)
            rep_by_length[seq_len].append(assigned_idx)

        assignments.append(
            ClusterAssignment(sequence=sequence, cluster_id=f"CL{assigned_idx + 1:06d}")
        )

    return assignments


def assign_cluster_splits(
    cluster_ids: list[str],
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
) -> dict[str, str]:
    """Assign cluster IDs to train/validation/test deterministically."""

    unique_clusters = sorted(set(cluster_ids))
    total = len(unique_clusters)
    train_end = int(total * train_fraction)
    validation_end = train_end + int(total * validation_fraction)
    split_map: dict[str, str] = {}
    for idx, cluster_id in enumerate(unique_clusters):
        if idx < train_end:
            split = "train"
        elif idx < validation_end:
            split = "validation"
        else:
            split = "test"
        split_map[cluster_id] = split
    return split_map
