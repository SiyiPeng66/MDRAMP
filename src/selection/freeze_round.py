"""Freeze a pre-label synthesis set and reserve order from a 200-proposal pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.selection.filters import enforce_pairwise_cosine, passes_guardrails
from src.selection.mixed_policy import select_mixed
from src.utils.hash import sha256_file
from src.optimization.pep_b_genetic import edit_distance


def next_reserve_candidate(reserve: pd.DataFrame, consumed_orders: set[int]) -> pd.Series:
    """Return only the next unconsumed item in the frozen reserve order."""
    if "reserve_order" not in reserve: raise ValueError("Reserve table lacks reserve_order")
    available = reserve[~reserve["reserve_order"].astype(int).isin(consumed_orders)].sort_values("reserve_order")
    if available.empty: raise LookupError("Frozen reserve list is exhausted")
    return available.iloc[0]


def freeze_round_selection(candidates: pd.DataFrame, embeddings: np.ndarray, *, budget: int, reserve_size: int, quotas: dict[str, float], reference_sequences=(), parent=None, parent_family=(), local_minimum_cosine=None):
    if len(candidates) != 200: raise ValueError(f"Paper proposal pool must contain exactly 200 rows; got {len(candidates)}")
    required = {"sequence", "R_t", "selection_category"}
    if not required.issubset(candidates): raise ValueError(f"Proposal pool missing {sorted(required - set(candidates))}")
    if len(embeddings) != len(candidates): raise ValueError("Embedding rows must align with proposal rows")
    if parent is not None:
        if parent not in parent_family: raise ValueError("Local family must include parent")
        reference_sequences = tuple(s for s in reference_sequences if s not in set(parent_family))
    elif parent_family or local_minimum_cosine is not None:
        raise ValueError("Local options require parent")
    if local_minimum_cosine is not None and not 0 <= local_minimum_cosine <= 2:
        raise ValueError("Invalid local cosine distance")
    eligibility = candidates["sequence"].astype(str).map(lambda seq: passes_guardrails(seq, reference_sequences=reference_sequences))
    eligible_mask = eligibility.map(lambda item: item[0]).to_numpy(copy=True)
    if parent is not None:
        eligible_mask &= candidates["sequence"].map(lambda s: abs(len(s)-len(parent)) <= 3 and 1 <= edit_distance(s,parent) <= min(int(.30*len(parent)),6)).to_numpy()
    eligible_mask &= ~candidates["sequence"].duplicated().to_numpy()
    eligible = candidates.loc[eligible_mask].copy()
    eligible_embeddings = np.asarray(embeddings)[eligible_mask]
    order = np.argsort(-eligible["R_t"].to_numpy(), kind="stable")
    eligible = eligible.iloc[order].copy()
    eligible_embeddings = eligible_embeddings[order]
    minimum = .10 if parent is None else local_minimum_cosine
    diverse = list(range(len(eligible))) if minimum is None else enforce_pairwise_cosine(eligible["sequence"].tolist(), eligible_embeddings, minimum=minimum)
    eligible = eligible.iloc[diverse].copy()
    if len(eligible) < budget + reserve_size: raise ValueError("Insufficient eligible diverse candidates for synthesis and reserve sets")
    chosen = select_mixed(eligible, budget=budget, quotas=quotas)
    remaining = eligible[~eligible["sequence"].isin(set(chosen["sequence"]))].sort_values("R_t", ascending=False)
    reserve = remaining.head(reserve_size).copy()
    chosen.insert(0, "selection_order", np.arange(1, len(chosen) + 1))
    reserve.insert(0, "reserve_order", np.arange(1, len(reserve) + 1))
    return chosen, reserve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--embeddings", required=True, help="Numpy matrix aligned to candidate CSV rows")
    parser.add_argument("--embedding-metadata", help="Metadata CSV for NPZ caches; aligns vectors by candidate ID")
    parser.add_argument("--id-column", default="peptide_id")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--reserve-size", type=int, required=True)
    parser.add_argument("--quotas-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-sequences", required=True, help="CSV sequence column: public training plus previously tested sequences")
    parser.add_argument("--parent")
    parser.add_argument("--parent-family", help="CSV sequence column of registered local family")
    parser.add_argument("--local-minimum-cosine", type=float, help="Explicit local diversity choice; no de novo default")
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    if args.embedding_metadata:
        from src.features.embedding_store import EmbeddingStore
        store = EmbeddingStore.load(args.embeddings,args.embedding_metadata)
        embeddings = store.global_by_record_id(candidates[args.id_column].astype(str).tolist())
        store.npz.close()
    else:
        embeddings = np.load(args.embeddings,allow_pickle=False)
        if not isinstance(embeddings,np.ndarray):
            raise ValueError("NPZ requires --embedding-metadata for ID alignment")
    quotas = json.loads(Path(args.quotas_json).read_text(encoding="utf-8"))
    if abs(sum(quotas.values()) - 1.0) > 1e-9: raise ValueError("Frozen selection quotas must sum to 1")
    references = pd.read_csv(args.reference_sequences)["sequence"].dropna().astype(str).tolist()
    if not references: raise ValueError("Homology reference must not be empty")
    family = () if not args.parent_family else pd.read_csv(args.parent_family)["sequence"].dropna().astype(str).tolist()
    selected, reserve = freeze_round_selection(candidates, embeddings, budget=args.budget, reserve_size=args.reserve_size, quotas=quotas, reference_sequences=references, parent=args.parent, parent_family=family, local_minimum_cosine=args.local_minimum_cosine)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected.csv", index=False); reserve.to_csv(output / "reserve.csv", index=False)
    manifest = {"embedding_metadata_sha256": sha256_file(args.embedding_metadata) if args.embedding_metadata else None, "parent": args.parent, "parent_family_sha256": sha256_file(args.parent_family) if args.parent_family else None, "local_minimum_cosine": args.local_minimum_cosine, "reference_sha256": sha256_file(args.reference_sequences), "candidate_pool_sha256": sha256_file(args.candidates), "embedding_sha256": sha256_file(args.embeddings), "quotas": quotas, "budget": args.budget, "reserve_size": args.reserve_size}
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__": main()
