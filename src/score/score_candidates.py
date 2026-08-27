"""Score candidates with the frozen paper Expert ensembles and MIC head."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import add_physicochemical_features, feature_columns
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import FeaturewiseGatedMembraneExpert, PaperAMPPriorExpert
from src.models.gated_integrator import GateParams, integrate_numpy
from src.models.mic_regressor import MICRegressionHead, activity_priority
from src.train.train_utils import require_file


def load_ensemble(path: str, expected: str) -> tuple[ExpertEnsemble, dict]:
    ckpt = torch.load(require_file(path), map_location="cpu")
    meta = ckpt["metadata"]
    if meta.get("architecture") != expected or len(meta.get("seeds", [])) != 5:
        raise ValueError(f"Paper mode requires a five-member {expected} checkpoint")
    if expected == "PaperAMPPriorExpert":
        factory = lambda: PaperAMPPriorExpert(meta["input_dim"], meta["branch_weight"])
    else:
        factory = lambda: FeaturewiseGatedMembraneExpert(meta["esm_dim"], meta["physchem_dim"])
    model = ExpertEnsemble(factory, seeds=meta["seeds"], output_key=meta["output_key"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="outputs/processed/private_R0.csv")
    parser.add_argument("--features", default="outputs/processed/candidate_features.csv")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--embedding-metadata", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--id-column", default="private_record_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--amp-prior", required=True)
    parser.add_argument("--membrane", required=True)
    parser.add_argument("--target-affinity", required=True)
    parser.add_argument("--mic-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/scores/candidate_scores.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--lambda-syn", type=float, default=0.3)
    args = parser.parse_args()

    add_physicochemical_features(Path(args.candidates), Path(args.features), sequence_column=args.sequence_column)
    frame = pd.read_csv(require_file(args.features))
    store = EmbeddingStore.load(args.embeddings, args.embedding_metadata)
    embeddings = store.global_by_record_id(frame[args.id_column].astype(str).tolist()).astype(np.float32)
    columns = feature_columns(frame)
    scaler = StateFeatureScaler.load(args.scaler)
    if scaler.columns != columns: raise ValueError("Candidate descriptors do not match the frozen scaler")
    physchem = scaler.transform(frame[columns].to_numpy(dtype=np.float32))
    matrix = np.concatenate([embeddings, physchem], axis=1)
    amp, _ = load_ensemble(args.amp_prior, "PaperAMPPriorExpert")
    membrane, _ = load_ensemble(args.membrane, "FeaturewiseGatedMembraneExpert")
    amp.to(args.device); membrane.to(args.device)
    with torch.no_grad():
        amp_output = amp(torch.as_tensor(matrix, device=args.device))
        mem_output = membrane(torch.as_tensor(embeddings, device=args.device), torch.as_tensor(physchem, device=args.device))
    prior_mean, prior_std = amp_output["mean"].cpu().numpy(), amp_output["std"].cpu().numpy()
    mem_mean, mem_std = mem_output["mean"].cpu().numpy(), mem_output["std"].cpu().numpy()

    affinity = pd.read_csv(require_file(args.target_affinity))
    required = {"peptide_id", "S_int_mean", "S_int_std", "top_target_id", "top_target_score"}
    if not required.issubset(affinity): raise ValueError(f"Interaction scores missing {sorted(required - set(affinity))}")
    merged = frame[[args.id_column, args.sequence_column]].merge(affinity[list(required)], left_on=args.id_column, right_on="peptide_id", validate="one_to_one")
    if len(merged) != len(frame): raise ValueError("Interaction scores do not cover every candidate")
    integrated = integrate_numpy(
        prior_mean, mem_mean, merged["S_int_mean"].to_numpy(),
        GateParams(args.tau, args.gamma, args.lambda_syn), prior_std, mem_std,
        merged["S_int_std"].to_numpy(), args.kappa,
    )
    mic_ckpt = torch.load(require_file(args.mic_checkpoint), map_location="cpu")
    mic_meta = mic_ckpt["metadata"]
    if not mic_meta.get("paper_prequential_validated"):
        raise ValueError("MIC checkpoint lacks a validated prequential state manifest")
    mic = MICRegressionHead(mic_meta["peptide_dim"], mic_meta["physchem_dim"])
    mic.load_state_dict(mic_ckpt["state_dict"]); mic.eval()
    means = np.column_stack([prior_mean, mem_mean, merged["S_int_mean"].to_numpy()]).astype(np.float32)
    with torch.no_grad():
        predicted_mic = mic(torch.as_tensor(embeddings), torch.as_tensor(physchem), torch.as_tensor(means)).numpy()
        activity = activity_priority(torch.as_tensor(predicted_mic)).numpy()
    output = merged[[args.id_column, args.sequence_column, "top_target_id", "top_target_score"]].copy()
    output["mu_prior"], output["sigma_prior"] = prior_mean, prior_std
    output["mu_mem"], output["sigma_mem"] = mem_mean, mem_std
    output["mu_int"], output["sigma_int"] = merged["S_int_mean"], merged["S_int_std"]
    output["S_prior"] = np.clip(prior_mean - args.kappa * prior_std, 0, 1)
    output["S_mem"] = np.clip(mem_mean - args.kappa * mem_std, 0, 1)
    output["S_int"] = np.clip(merged["S_int_mean"] - args.kappa * merged["S_int_std"], 0, 1)
    for key, value in integrated.items(): output[key] = value
    output["predicted_log2_mic"], output["A_t"] = predicted_mic, activity
    output["R_t"] = output["p_cons"] * output["A_t"]
    output = output.sort_values("R_t", ascending=False).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output, index=False)


if __name__ == "__main__": main()
