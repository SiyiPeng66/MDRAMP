"""Paper-faithful public initialization of AMP and membrane Expert ensembles."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import FeaturewiseGatedMembraneExpert, PaperAMPPriorExpert
from src.train.expert_losses import amp_prior_loss, masked_membrane_loss
from src.utils.hash import hash_existing
from src.train.train_utils import require_file, save_checkpoint, save_json

SEEDS = (11, 23, 37, 51, 73)
AUX_COLUMNS = ("membrane_disruption", "pore_forming", "permeabilization", "depolarization")


def load_domain(feature_csv: str, embedding_npz: str, metadata_csv: str, id_column: str):
    frame = pd.read_csv(require_file(feature_csv))
    store = EmbeddingStore.load(require_file(embedding_npz), require_file(metadata_csv))
    embeddings = store.global_by_record_id(frame[id_column].astype(str).tolist()).astype(np.float32)
    columns = feature_columns(frame)
    return frame, embeddings, frame[columns].to_numpy(dtype=np.float32), columns


def train_amp_ensemble(ensemble, positive, unlabelled, *, epochs, batch_size, learning_rate, positive_prior, manifold_weight, device, weight_decay=0.01):
    histories = []
    for seed, member in zip(ensemble.seeds, ensemble.members):
        print(f"AMP member seed={seed} starting epochs={epochs}",flush=True)
        torch.manual_seed(seed)
        member.to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        history = []
        for epoch in range(epochs):
            pos_order = torch.randperm(len(positive), generator=generator)
            unl_order = torch.randperm(len(unlabelled), generator=generator)
            losses = []
            steps = max(int(np.ceil(len(positive) / batch_size)), int(np.ceil(len(unlabelled) / batch_size)))
            for step in range(steps):
                pi = pos_order[(step * batch_size) % len(pos_order):][:batch_size]
                ui = unl_order[(step * batch_size) % len(unl_order):][:batch_size]
                if not len(pi): pi = pos_order[:batch_size]
                if not len(ui): ui = unl_order[:batch_size]
                pos_out = member.discriminative(torch.as_tensor(positive[pi], dtype=torch.float32, device=device))
                unl_out = member.discriminative(torch.as_tensor(unlabelled[ui], dtype=torch.float32, device=device))
                loss = amp_prior_loss(pos_out, unl_out, positive_prior=positive_prior, manifold_weight=manifold_weight)
                if not torch.isfinite(loss): raise ValueError(f"Nonfinite AMP loss: seed={seed}, epoch={epoch+1}")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(float(np.mean(losses)))
            if (epoch+1)%10==0 or epoch+1==epochs:
                print(f"AMP seed={seed} epoch={epoch+1}/{epochs} loss={history[-1]:.6f}",flush=True)
        histories.append(history)
    return histories


def train_membrane_ensemble(ensemble, embeddings, physchem, primary, auxiliary, *, epochs, batch_size, learning_rate, teacher_weight, device, weight_decay=0.01):
    primary_mask = np.isfinite(primary) & (primary >= 0) & (primary <= 1) & ((primary <= .2) | (primary >= .8))
    auxiliary_mask = np.zeros_like(auxiliary, dtype=bool)
    primary, auxiliary = np.nan_to_num(primary).astype(np.float32), np.nan_to_num(auxiliary).astype(np.float32)
    if not primary_mask.any() and not auxiliary_mask.any():
        raise ValueError("No membrane supervision is available")
    histories = []
    for seed, member in zip(ensemble.seeds, ensemble.members):
        print(f"Membrane member seed={seed} starting epochs={epochs}",flush=True)
        torch.manual_seed(seed)
        member.to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        history = []
        for epoch in range(epochs):
            losses = []
            for idx in torch.randperm(len(primary), generator=generator).split(batch_size):
                ix = idx.numpy()
                if not primary_mask[ix].any() and not auxiliary_mask[ix].any():
                    continue
                outputs = member(torch.as_tensor(embeddings[ix], device=device), torch.as_tensor(physchem[ix], device=device))
                loss = masked_membrane_loss(
                    outputs, torch.as_tensor(primary[ix], device=device), torch.as_tensor(primary_mask[ix], device=device),
                    torch.as_tensor(auxiliary[ix], device=device), torch.as_tensor(auxiliary_mask[ix], device=device),
                    teacher_weight=teacher_weight,
                )
                if not torch.isfinite(loss): raise ValueError(f"Nonfinite membrane loss: seed={seed}, epoch={epoch+1}")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise ValueError("Epoch contains no supervised membrane batches")
            history.append(float(np.mean(losses)))
            if (epoch+1)%10==0 or epoch+1==epochs:
                print(f"Membrane seed={seed} epoch={epoch+1}/{epochs} loss={history[-1]:.6f}",flush=True)
        histories.append(history)
    return histories


def fit_ensemble_support(ensemble, embeddings, clusters):
    """Calibrate identical nonparametric support once for all five members."""
    print(f"Calibrating frozen AMP support: references={len(embeddings)}",flush=True)
    first=ensemble.members[0]
    first.fit_support(embeddings,clusters)
    for member in ensemble.members[1:]:
        for name in ("reference","reference_clusters","radius","spread"):
            setattr(member,name,getattr(first,name).clone())
    print("Frozen AMP support calibrated and copied to ensemble",flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-features", default="outputs/processed/public_features.csv")
    parser.add_argument("--positive-embeddings", default="outputs/esm3_embeddings/public_peptides.npz")
    parser.add_argument("--positive-embedding-metadata", default="outputs/esm3_embeddings/public_peptides_metadata.csv")
    parser.add_argument("--background-features", required=True)
    parser.add_argument("--background-embeddings", required=True)
    parser.add_argument("--background-embedding-metadata", required=True)
    parser.add_argument("--id-column", default="sequence")
    parser.add_argument("--pore-scores", required=True)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/M_R0")
    parser.add_argument("--scaler-output", default="outputs/checkpoints/M_R0/feature_scaler.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--positive-prior", type=float, default=0.10)
    parser.add_argument("--manifold-loss-weight", type=float, default=0.0)
    parser.add_argument("--amp-branch-weight", type=float, default=0.25, help="Nonparametric support mixing eta (V6: 0.25)")
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ensemble-seeds", nargs=5, type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid optimization hyperparameters")
    if args.manifold_loss_weight != 0:
        parser.error("V6 has no manifold training loss; --manifold-loss-weight must be zero")
    if not 0 <= args.amp_branch_weight <= 1 or not 0 < args.positive_prior < 1 or args.teacher_weight <= 0:
        parser.error("Invalid expert objective hyperparameters")
    if len(set(args.ensemble_seeds)) != 5:
        parser.error("Five independent seeds required")

    pos, pos_emb, pos_pc, columns = load_domain(args.positive_features, args.positive_embeddings, args.positive_embedding_metadata, args.id_column)
    if "split" in pos:
        mask = pos["split"].astype(str).eq("train").to_numpy()
        pos, pos_emb, pos_pc = pos.loc[mask].reset_index(drop=True), pos_emb[mask], pos_pc[mask]
    background, bg_emb, bg_pc, bg_columns = load_domain(args.background_features, args.background_embeddings, args.background_embedding_metadata, args.id_column)
    if columns != bg_columns: raise ValueError("Positive and background descriptor columns differ")
    if len(pos) != len(background): raise ValueError("Paper mode requires a 1:1 positive/unlabelled AMP set")
    scaler = StateFeatureScaler.fit(np.concatenate([pos_pc, bg_pc]), columns)
    scaler.save(args.scaler_output)
    pos_pc, bg_pc = scaler.transform(pos_pc), scaler.transform(bg_pc)
    positive = np.concatenate([pos_emb, pos_pc], axis=1)
    unlabelled = np.concatenate([bg_emb, bg_pc], axis=1)
    checkpoint_dir = Path(args.checkpoint_dir)

    amp = ExpertEnsemble(lambda: PaperAMPPriorExpert(positive.shape[1], args.amp_branch_weight), seeds=args.ensemble_seeds, output_key="probability")
    if "cluster_id" not in pos: raise ValueError("Training AMP cluster_id is required for support calibration")
    fit_ensemble_support(amp, pos_emb, pos["cluster_id"].tolist())
    amp_losses = train_amp_ensemble(amp, positive, unlabelled, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, positive_prior=args.positive_prior, manifold_weight=args.manifold_loss_weight, device=args.device, weight_decay=args.weight_decay)
    save_checkpoint(amp, checkpoint_dir / "amp_prior_ensemble.pt", metadata={
        "training_config": vars(args),
        "input_hashes": hash_existing([args.positive_features,args.positive_embeddings,args.background_features,args.background_embeddings]),
        "architecture": "PaperAMPPriorExpert", "input_dim": positive.shape[1], "esm_dim": pos_emb.shape[1],
        "physchem_dim": pos_pc.shape[1], "seeds": list(args.ensemble_seeds), "output_key": "probability",
        "objective": "nnPU_with_nonparametric_support", "background_label_type": "unlabelled", "losses": amp_losses,
        "positive_prior": args.positive_prior, "manifold_loss_weight": args.manifold_loss_weight,
        "branch_weight": args.amp_branch_weight,
    })

    teacher = pd.read_csv(require_file(args.pore_scores))
    teacher_key = "record_id" if "record_id" in teacher else args.id_column
    supervised = pos[[args.id_column]].merge(teacher, left_on=args.id_column, right_on=teacher_key, how="left")
    if "pore_teacher_score" not in supervised: raise ValueError("Pore scores require pore_teacher_score")
    auxiliary = np.column_stack([pd.to_numeric(supervised[c], errors="coerce").to_numpy(dtype=np.float32) if c in supervised else np.full(len(pos), np.nan, np.float32) for c in AUX_COLUMNS])
    membrane = ExpertEnsemble(lambda: FeaturewiseGatedMembraneExpert(pos_emb.shape[1], pos_pc.shape[1]), seeds=args.ensemble_seeds, output_key="primary")
    mem_losses = train_membrane_ensemble(membrane, pos_emb, pos_pc, pd.to_numeric(supervised["pore_teacher_score"], errors="coerce").to_numpy(dtype=np.float32), auxiliary, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, teacher_weight=args.teacher_weight, device=args.device, weight_decay=args.weight_decay)
    save_checkpoint(membrane, checkpoint_dir / "membrane_ensemble.pt", metadata={
        "training_config": vars(args), "teacher_sha256": hash_existing([args.pore_scores]),
        "architecture": "FeaturewiseGatedMembraneExpert", "esm_dim": pos_emb.shape[1], "physchem_dim": pos_pc.shape[1],
        "seeds": list(args.ensemble_seeds), "output_key": "primary", "auxiliary_columns": list(AUX_COLUMNS),
        "objective": "single_task_teacher_BCE", "losses": mem_losses,
    })
    save_json({"state": "M_R0", "status": "paper_expert_ensembles_complete"}, checkpoint_dir / "metadata.json")


if __name__ == "__main__": main()
