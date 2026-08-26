"""Score peptide candidates with trained expert checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.features.physicochemical import add_physicochemical_features
from src.models.expert_modules import AMPPriorExpert, MembraneMechanismExpert
from src.models.gated_integrator import GateParams, integrate_numpy
from src.train.train_utils import load_peptide_matrix, require_file
from src.models.mic_regressor import MICRegressionHead, activity_priority


def load_expert(path: Path, cls):
    ckpt = torch.load(require_file(path), map_location="cpu")
    model = cls(input_dim=ckpt["metadata"]["input_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model: torch.nn.Module, matrix, device: str):
    model.to(device)
    with torch.no_grad():
        return model(torch.tensor(matrix, dtype=torch.float32, device=device)).detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="outputs/processed/private_R0.csv")
    parser.add_argument("--features", default="outputs/processed/candidate_features.csv")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--embedding-metadata", required=True)
    parser.add_argument("--id-column", default="private_record_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--amp-prior", default="outputs/checkpoints/M_R0/amp_prior.pt")
    parser.add_argument("--membrane", default="outputs/checkpoints/M_R1/membrane_expert.pt")
    parser.add_argument("--target-affinity", required=True, help="Aggregated TPepPro/model S_aff CSV.")
    parser.add_argument("--mic-checkpoint", default=None)
    parser.add_argument("--output", default="outputs/scores/candidate_scores.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--lambda-syn", type=float, default=0.3)
    args = parser.parse_args()

    add_physicochemical_features(Path(args.candidates), Path(args.features), sequence_column=args.sequence_column)
    df, matrix = load_peptide_matrix(
        Path(args.features),
        Path(args.embeddings),
        Path(args.embedding_metadata),
        id_column=args.id_column,
    )
    amp = load_expert(Path(args.amp_prior), AMPPriorExpert)
    mem = load_expert(Path(args.membrane), MembraneMechanismExpert)
    S_prior = predict(amp, matrix, args.device)
    S_mem = predict(mem, matrix, args.device)

    aff = pd.read_csv(require_file(args.target_affinity))
    aff_key = "peptide_id" if "peptide_id" in aff.columns else args.id_column
    merged = df[[args.id_column, args.sequence_column]].merge(
        aff[[aff_key, "S_aff", "top_target_id", "top_target_score"]],
        left_on=args.id_column,
        right_on=aff_key,
        how="left",
    )
    if merged["S_aff"].isna().any():
        raise ValueError("Target affinity file does not cover all candidate IDs.")
    integrated = integrate_numpy(
        S_prior,
        S_mem,
        merged["S_aff"].to_numpy(),
        GateParams(tau=args.tau, gamma=args.gamma, lambda_syn=args.lambda_syn),
    )
    if args.mic_checkpoint:
        mic_ckpt = torch.load(require_file(args.mic_checkpoint), map_location="cpu")
        mic_meta = mic_ckpt["metadata"]
        mic = MICRegressionHead(
            peptide_dim=mic_meta["peptide_dim"],
            physchem_dim=mic_meta["physchem_dim"],
        )
        mic.load_state_dict(mic_ckpt["state_dict"])
        mic.eval()
        embedding_dim = mic_meta["peptide_dim"]
        pc = torch.tensor(matrix[:, embedding_dim:], dtype=torch.float32)
        pep = torch.tensor(matrix[:, :embedding_dim], dtype=torch.float32)
        means = torch.tensor(
            pd.DataFrame({"prior": S_prior, "mem": S_mem, "aff": merged["S_aff"]}).to_numpy(),
            dtype=torch.float32,
        )
        with torch.no_grad():
            predicted_mic = mic(pep, pc, means).numpy()
            activity = activity_priority(torch.tensor(predicted_mic)).numpy()
        deployed_score = integrated["p_cons"] * activity
    else:
        predicted_mic = None
        activity = None
        deployed_score = integrated["p_cons"]
    out = pd.DataFrame(
        {
            "sequence": df[args.sequence_column],
            "S_prior": S_prior,
            "S_mem": S_mem,
            "S_aff": merged["S_aff"],
            "g_prior": integrated["g_prior"],
            "E_comp": integrated["E_comp"],
            "E_syn": integrated["E_syn"],
            "E_mech": integrated["E_mech"],
            "p_cons": integrated["p_cons"],
            "predicted_log2_mic": predicted_mic,
            "A_t": activity,
            "R_t": deployed_score,
            "top_target_id": merged["top_target_id"],
            "top_target_score": merged["top_target_score"],
        }
    )
    out.insert(0, args.id_column, df[args.id_column])
    out = out.sort_values("R_t", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"candidate scores: {out.shape} -> {args.output}")


if __name__ == "__main__":
    main()
