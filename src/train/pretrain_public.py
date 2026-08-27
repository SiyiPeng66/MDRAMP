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
from src.train.train_utils import require_file, save_checkpoint, save_json

SEEDS = (11, 23, 37, 51, 73)
AUX_COLUMNS = ("membrane_disruption", "pore_forming", "permeabilization", "depolarization")


def load_domain(feature_csv: str, embedding_npz: str, metadata_csv: str, id_column: str):
    frame = pd.read_csv(require_file(feature_csv))
    store = EmbeddingStore.load(require_file(embedding_npz), require_file(metadata_csv))
    embeddings = store.global_by_record_id(frame[id_column].astype(str).tolist()).astype(np.float32)
    columns = feature_columns(frame)
    return frame, embeddings, frame[columns].to_numpy(dtype=np.float32), columns


def train_amp_ensemble(ensemble, positive, unlabelled, *, epochs, batch_size, learning_rate, positive_prior, manifold_weight, device):
    histories = []
    for seed, member in zip(ensemble.seeds, ensemble.members):
        member.to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        history = []
        for _ in range(epochs):
            pos_order = torch.randperm(len(positive), generator=generator)
            unl_order = torch.randperm(len(unlabelled), generator=generator)
            losses = []
            steps = max(int(np.ceil(len(positive) / batch_size)), int(np.ceil(len(unlabelled) / batch_size)))
            for step in range(steps):
                pi = pos_order[(step * batch_size) % len(pos_order):][:batch_size]
                ui = unl_order[(step * batch_size) % len(unl_order):][:batch_size]
                if not len(pi): pi = pos_order[:batch_size]
                if not len(ui): ui = unl_order[:batch_size]
                pos_out = member(torch.as_tensor(positive[pi], dtype=torch.float32, device=device))
                unl_out = member(torch.as_tensor(unlabelled[ui], dtype=torch.float32, device=device))
                loss = amp_prior_loss(pos_out, unl_out, positive_prior=positive_prior, manifold_weight=manifold_weight)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append(float(np.mean(losses)))
        histories.append(history)
    return histories


def train_membrane_ensemble(ensemble, embeddings, physchem, primary, auxiliary, *, epochs, batch_size, learning_rate, teacher_weight, device):
    primary_mask, auxiliary_mask = ~np.isnan(primary), ~np.isnan(auxiliary)
    primary, auxiliary = np.nan_to_num(primary).astype(np.float32), np.nan_to_num(auxiliary).astype(np.float32)
    if not primary_mask.any() and not auxiliary_mask.any():
        raise ValueError("No membrane supervision is available")
    histories = []
    for seed, member in zip(ensemble.seeds, ensemble.members):
        member.to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        history = []
        for _ in range(epochs):
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
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise ValueError("Epoch contains no supervised membrane batches")
            history.append(float(np.mean(losses)))
        histories.append(history)
    return histories


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--positive-prior", type=float, required=True)
    parser.add_argument("--manifold-loss-weight", type=float, required=True)
    parser.add_argument("--amp-branch-weight", type=float, required=True, help="Weight assigned to the nnPU branch in the final AMP score")
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    args = parser.parse_args()

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

    amp = ExpertEnsemble(lambda: PaperAMPPriorExpert(positive.shape[1], args.amp_branch_weight), seeds=SEEDS, output_key="probability")
    amp_losses = train_amp_ensemble(amp, positive, unlabelled, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, positive_prior=args.positive_prior, manifold_weight=args.manifold_loss_weight, device=args.device)
    save_checkpoint(amp, checkpoint_dir / "amp_prior_ensemble.pt", metadata={
        "architecture": "PaperAMPPriorExpert", "input_dim": positive.shape[1], "esm_dim": pos_emb.shape[1],
        "physchem_dim": pos_pc.shape[1], "seeds": list(SEEDS), "output_key": "probability",
        "objective": "nnPU_plus_manifold", "background_label_type": "unlabelled", "losses": amp_losses,
        "positive_prior": args.positive_prior, "manifold_loss_weight": args.manifold_loss_weight,
        "branch_weight": args.amp_branch_weight,
    })

    teacher = pd.read_csv(require_file(args.pore_scores))
    teacher_key = "record_id" if "record_id" in teacher else args.id_column
    supervised = pos[[args.id_column]].merge(teacher, left_on=args.id_column, right_on=teacher_key, how="left")
    if "pore_teacher_score" not in supervised: raise ValueError("Pore scores require pore_teacher_score")
    auxiliary = np.column_stack([pd.to_numeric(supervised[c], errors="coerce").to_numpy(dtype=np.float32) if c in supervised else np.full(len(pos), np.nan, np.float32) for c in AUX_COLUMNS])
    membrane = ExpertEnsemble(lambda: FeaturewiseGatedMembraneExpert(pos_emb.shape[1], pos_pc.shape[1]), seeds=SEEDS, output_key="primary")
    mem_losses = train_membrane_ensemble(membrane, pos_emb, pos_pc, pd.to_numeric(supervised["pore_teacher_score"], errors="coerce").to_numpy(dtype=np.float32), auxiliary, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, teacher_weight=args.teacher_weight, device=args.device)
    save_checkpoint(membrane, checkpoint_dir / "membrane_ensemble.pt", metadata={
        "architecture": "FeaturewiseGatedMembraneExpert", "esm_dim": pos_emb.shape[1], "physchem_dim": pos_pc.shape[1],
        "seeds": list(SEEDS), "output_key": "primary", "auxiliary_columns": list(AUX_COLUMNS),
        "objective": "masked_multitask_BCE_plus_teacher_soft_supervision", "losses": mem_losses,
    })
    save_json({"state": "M_R0", "status": "paper_expert_ensembles_complete"}, checkpoint_dir / "metadata.json")


if __name__ == "__main__": main()
