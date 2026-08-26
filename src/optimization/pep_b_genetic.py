"""Constrained Pep-B local sequence optimizer."""

from __future__ import annotations

import random
from collections.abc import Callable


def edit_distance(a: str, b: str) -> int:
    rows = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, rows[j] + 1, rows[j - 1] + (x != y)))
        rows = current
    return rows[-1]


def mutate(sequence: str, rng: random.Random, amino_acids: str = "ACDEFGHIKLMNPQRSTVWY") -> str:
    chars = list(sequence)
    event = rng.choices(["substitution", "insertion", "deletion"], weights=[0.8, 0.1, 0.1])[0]
    idx = rng.randrange(len(chars))
    if event == "substitution":
        chars[idx] = rng.choice(amino_acids)
    elif event == "insertion" and len(chars) < 50:
        chars.insert(idx, rng.choice(amino_acids))
    elif event == "deletion" and len(chars) > 10:
        chars.pop(idx)
    return "".join(chars)


def optimize(
    parent: str,
    fitness: Callable[[str], float],
    *,
    seed: int,
    population_size: int = 200,
    generations: int = 100,
) -> tuple[str, float]:
    rng = random.Random(seed)
    population = [parent] + [mutate(parent, rng) for _ in range(population_size - 1)]
    best, best_score, stale = parent, float("-inf"), 0
    for _ in range(generations):
        scored = sorted(((fitness(seq), seq) for seq in population), reverse=True)
        score, seq = scored[0]
        if score - best_score < 0.001:
            stale += 1
        else:
            best, best_score, stale = seq, score, 0
        if stale >= 15:
            break
        elites = [item[1] for item in scored[: max(1, population_size // 10)]]
        population = elites[:]
        while len(population) < population_size:
            base = rng.choice(elites)
            edits = rng.choices([1, 2, 3], weights=[0.50, 0.35, 0.15])[0]
            child = base
            for _ in range(edits):
                child = mutate(child, rng)
            if edit_distance(parent, child) <= min(int(0.30 * len(parent)), 6) and abs(len(child) - len(parent)) <= 3:
                population.append(child)
    return best, best_score
