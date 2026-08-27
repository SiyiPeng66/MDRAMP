"""Manuscript-parameterized constrained Pep-B genetic optimization."""
from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from src.selection.filters import passes_guardrails

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAPER_SEEDS = (101, 211, 307, 401, 503)


@dataclass(frozen=True)
class FitnessComponents:
    consistency: float
    novelty: float
    diversity: float

    @property
    def weighted(self) -> float:
        return 0.60 * self.consistency + 0.20 * self.novelty + 0.20 * self.diversity


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 200
    generations: int = 100
    tournament_size: int = 3
    elitism_fraction: float = 0.10
    crossover_probability: float = 0.60
    mutation_probability: float = 0.80
    patience: int = 15
    minimum_improvement: float = 0.001


def edit_distance(a: str, b: str) -> int:
    rows = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, rows[j] + 1, rows[j - 1] + (x != y)))
        rows = current
    return rows[-1]


def mutate(sequence: str, rng: random.Random, amino_acids: str = AMINO_ACIDS) -> str:
    chars = list(sequence)
    event = rng.choices(("substitution", "insertion", "deletion"), weights=(0.8, 0.1, 0.1))[0]
    idx = rng.randrange(len(chars))
    if event == "substitution": chars[idx] = rng.choice(amino_acids)
    elif event == "insertion" and len(chars) < 50: chars.insert(idx, rng.choice(amino_acids))
    elif event == "deletion" and len(chars) > 10: chars.pop(idx)
    return "".join(chars)


def two_point_crossover(left: str, right: str, rng: random.Random) -> tuple[str, str]:
    if len(left) != len(right) or len(left) < 3: return left, right
    start, end = sorted(rng.sample(range(1, len(left)), 2))
    return left[:start] + right[start:end] + left[end:], right[:start] + left[start:end] + right[end:]


def _score(value: FitnessComponents | tuple[float, float, float] | dict[str, float]) -> float:
    if isinstance(value, FitnessComponents): return value.weighted
    if isinstance(value, tuple) and len(value) == 3: return FitnessComponents(*map(float, value)).weighted
    if isinstance(value, dict): return FitnessComponents(float(value["consistency"]), float(value["novelty"]), float(value["diversity"])).weighted
    raise TypeError("Paper GA fitness must expose consistency, novelty and diversity components")


def _eligible(parent: str, candidate: str, reference_sequences: Iterable[str]) -> bool:
    if abs(len(candidate) - len(parent)) > 3: return False
    if edit_distance(parent, candidate) > min(int(0.30 * len(parent)), 6): return False
    return passes_guardrails(candidate, reference_sequences=reference_sequences)[0]


def optimize(
    parent: str,
    fitness: Callable[[str], FitnessComponents | tuple[float, float, float] | dict[str, float]],
    *,
    seed: int,
    config: GAConfig = GAConfig(),
    reference_sequences: Iterable[str] = (),
) -> tuple[str, float]:
    rng = random.Random(seed)
    references = tuple(reference_sequences)
    valid, reason = passes_guardrails(parent)
    if not valid: raise ValueError(f"Pep-B parent fails guardrails: {reason}")
    population = [parent]
    while len(population) < config.population_size:
        child = mutate(parent, rng)
        if _eligible(parent, child, references): population.append(child)
    best, best_score, stale = parent, _score(fitness(parent)), 0
    elite_count = max(1, int(config.population_size * config.elitism_fraction))
    for _ in range(config.generations):
        scored = sorted(((_score(fitness(seq)), seq) for seq in population), reverse=True)
        generation_score, generation_best = scored[0]
        if generation_score - best_score >= config.minimum_improvement:
            best, best_score, stale = generation_best, generation_score, 0
        else:
            if generation_score > best_score: best, best_score = generation_best, generation_score
            stale += 1
        if stale >= config.patience: break
        score_by_sequence = dict((seq, score) for score, seq in scored)
        def tournament() -> str:
            contenders = rng.sample(population, config.tournament_size)
            return max(contenders, key=score_by_sequence.__getitem__)
        next_population = [seq for _, seq in scored[:elite_count]]
        attempts = 0
        while len(next_population) < config.population_size and attempts < config.population_size * 100:
            attempts += 1
            left, right = tournament(), tournament()
            children = two_point_crossover(left, right, rng) if rng.random() < config.crossover_probability else (left, right)
            for child in children:
                if rng.random() < config.mutation_probability:
                    for _ in range(rng.choices((1, 2, 3), weights=(0.50, 0.35, 0.15))[0]): child = mutate(child, rng)
                if _eligible(parent, child, references) and child not in next_population:
                    next_population.append(child)
                if len(next_population) == config.population_size: break
        if len(next_population) < config.population_size:
            next_population.extend(seq for _, seq in scored if seq not in next_population)
        population = next_population[:config.population_size]
    return best, best_score


def optimize_five_runs(parent: str, fitness: Callable, *, reference_sequences: Iterable[str] = (), config: GAConfig = GAConfig()):
    return [optimize(parent, fitness, seed=seed, config=config, reference_sequences=reference_sequences) for seed in PAPER_SEEDS]
