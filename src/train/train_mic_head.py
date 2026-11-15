"""Train the lightweight, state-specific quantitative MIC regression head."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.features.scaler import StateFeatureScaler
from src.loop.prequential_guard import assert_no_current_round_labels, assert_round_cutoff
from src.loop.state_manifest import load_manifest
from src.models.mic_regressor import MICRegressionHead, censored_huber_loss
from src.train.train_utils import require_file, save_checkpoint
from src.utils.hash import sha256_file



FROZEN_ARTIFACTS = ('esm_checkpoint', 'scaler', 'amp_prior', 'membrane', 'target_affinity',
                    'target_embeddings', 'target_embedding_metadata', 'panel_manifest')


def validate_measured_targets(ledger):
    """S8 admits measured panel-best exact MIC or >256 right censoring only."""
    required = {'target_semantics', 'target_definition', 'mic_units', 'measurement_type'}
    if not required.issubset(ledger):
        raise ValueError(f'Measured MIC provenance columns required: {sorted(required - set(ledger))}')
    if not ledger.target_semantics.eq('measured_mic').all() or not ledger.target_definition.eq('panel_best_mic').all() or not ledger.mic_units.eq('ug/mL').all():
        raise ValueError('Only experimentally measured panel-best MIC in ug/mL is accepted')
    if any(str(c).startswith('measured_') for c in ledger) or ('target_divisor' in ledger and not pd.to_numeric(ledger.target_divisor).eq(1).all()):
        raise ValueError('Artificial/scaled MIC targets are not paper training labels')
    if not ledger.measurement_type.isin(['exact', 'right_censored']).all():
        raise ValueError('Left/interval censored screening labels are unsupported by S8 loss')
    flags = ledger.mic_censored.astype(str).str.lower()
    if not flags.isin(['true','false','0','1']).all(): raise ValueError('Invalid censor flags')
    censored = flags.isin(['true','1'])
    if not censored.eq(ledger.measurement_type.eq('right_censored')).all():
        raise ValueError('Measurement type and censor flag disagree')
    targets = pd.to_numeric(ledger.mic_log2, errors='raise')
    if not np.isfinite(targets).all() or not targets[censored].eq(8).all() or targets[~censored].gt(8).any():
        raise ValueError('MIC targets must respect the measured assay range and >256 boundary 8')
    return censored


def bound_artifacts(manifest, args):
    """Require an explicit immutable snapshot, not a self-declared validation flag."""
    paths = {key: str(require_file(getattr(args, key))) for key in
             ('mic_ledger','peptide_features','peptide_embeddings','peptide_embedding_metadata','expert_means','scaler')}
    frozen = manifest.config.get('frozen_scorer_artifacts', {})
    for key in FROZEN_ARTIFACTS:
        if key not in frozen: raise ValueError(f'Manifest frozen_scorer_artifacts missing {key}')
        artifact = Path(frozen[key])
        if not artifact.is_absolute():
            if not getattr(args, 'state_manifest', None): raise ValueError('Relative frozen artifact requires state_manifest path')
            artifact = Path(args.state_manifest).resolve().parent / artifact
        paths[key] = str(require_file(artifact.resolve()))
    if sha256_file(args.scaler) != sha256_file(paths['scaler']): raise ValueError('State scaler mismatch')
    hashes = {key:sha256_file(path) for key,path in paths.items()}
    if any(manifest.input_hashes.get(key) != digest for key,digest in hashes.items()):
        raise ValueError('State input_hashes must bind every MIC input and frozen scoring artifact by argument name')
    return paths, hashes

def train_mic_head(
    model: MICRegressionHead,
    peptide_embedding: np.ndarray,
    physchem: np.ndarray,
    expert_means: np.ndarray,
    target: np.ndarray,
    censored: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    device: str = "cpu",
    validation: tuple | None = None,
    weight_decay: float = 1e-4,
    delta: float = 1.0,
    patience: int = 15,
) -> list[float]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pep = torch.tensor(peptide_embedding, dtype=torch.float32)
    pc = torch.tensor(physchem, dtype=torch.float32)
    means = torch.tensor(expert_means, dtype=torch.float32)
    yy = torch.tensor(target, dtype=torch.float32)
    cc = torch.tensor(censored, dtype=torch.bool)
    losses = []
    best_loss, best_state = float("inf"), None
    stale = 0
    for epoch in range(epochs):
        order = torch.randperm(len(yy))
        batch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pred = model(pep[idx].to(device), pc[idx].to(device), means[idx].to(device))
            loss = censored_huber_loss(pred, yy[idx].to(device), cc[idx].to(device), delta=delta)
            if not torch.isfinite(loss): raise ValueError("Nonfinite MIC training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
        if validation is not None:
            model.eval()
            with torch.no_grad():
                vp, vc, vm, vy, vz = [torch.tensor(v, device=device) for v in validation]
                value = float(censored_huber_loss(model(vp.float(), vc.float(), vm.float()), vy.float(), vz.bool(), delta=delta))
            if not np.isfinite(value): raise ValueError("Nonfinite MIC validation loss")
            if value < best_loss:
                best_loss, best_state = value, copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            model.train()
            if stale >= patience: break
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"MIC epoch {epoch + 1}: loss={losses[-1]:.6g}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic-ledger", required=True)
    parser.add_argument("--peptide-features", required=True)
    parser.add_argument("--peptide-embeddings", required=True)
    parser.add_argument("--peptide-embedding-metadata", required=True)
    parser.add_argument("--expert-means", required=True, help="Frozen ensemble means keyed by peptide_id: mu_prior,mu_mem,mu_int.")
    parser.add_argument("--state-manifest", required=True)
    parser.add_argument("--current-round", required=True, choices=[f"R{i}" for i in range(6)])
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--peptide-id-column", default="private_record_id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (200 if args.current_round == "R0" else 100)
    args.learning_rate = args.learning_rate if args.learning_rate is not None else (1e-4 if args.current_round == "R0" else 5e-5)
    if args.patience < 1 or args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or args.huber_delta <= 0:
        parser.error("Invalid optimization hyperparameters")
    if args.huber_delta != 1: parser.error("S8 Huber delta is fixed at 1")
    if Path(args.output).exists(): raise ValueError("MIC output already exists")
    torch.manual_seed(args.seed)
    if args.current_round != "R0" and not args.parent_checkpoint:
        parser.error("Subsequent states require --parent-checkpoint")

    ledger = pd.read_csv(require_file(args.mic_ledger))
    manifest = load_manifest(require_file(args.state_manifest))
    manifest.assert_paper_policy()
    input_paths, input_hashes = bound_artifacts(manifest, args)
    input_paths["state_manifest"] = args.state_manifest
    input_hashes["state_manifest"] = sha256_file(args.state_manifest)
    validate_measured_targets(ledger)
    expected_parent = None if args.current_round == "R0" else f"M_R{int(args.current_round[1:])-1}"
    if manifest.parent_state != expected_parent:
        raise ValueError("Manifest must reference the immediately preceding state")
    if manifest.state_name != f"M_{args.current_round}" or manifest.visible_round != args.current_round:
        raise ValueError("Manifest state must match current round")
    assert_round_cutoff(ledger[ledger["round"] != "INIT"] if "round" in ledger else ledger, visible_round=manifest.visible_round)
    assert_no_current_round_labels(
        ledger, current_round=args.current_round,
        label_columns=("mic_log2", "mic_censored", "label_4class", "endpoint_256"),
    )
    required = {"peptide_id", "mic_log2", "mic_censored", "round", "split", "released_at", "qc_pass"}
    if not required.issubset(ledger.columns):
        raise ValueError(f"MIC ledger missing columns: {sorted(required - set(ledger.columns))}")
    release = pd.to_datetime(ledger["released_at"], utc=True, errors="raise")
    cutoff = pd.Timestamp(manifest.data_cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Manifest cutoff must include timezone")
    if release.isna().any() or (release > cutoff).any() or ledger["qc_pass"].astype(str).str.lower().ne("true").any():
        raise ValueError("Unreleased or non-QC-passed MIC records")
    if ledger["peptide_id"].duplicated().any():
        raise ValueError("Duplicate MIC identifiers; aggregate replicates before training")
    if not set(ledger["split"]).issubset({"train", "validation"}):
        raise ValueError("Only explicit train/validation records allowed")
    flags = ledger["mic_censored"].astype(str).str.lower()
    if not flags.isin(["true", "false", "0", "1"]).all():
        raise ValueError("Invalid censor flags")
    ledger["mic_censored"] = flags.isin(["true", "1"])
    if not np.isfinite(ledger["mic_log2"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite MIC targets")
    if not ledger.loc[ledger["mic_censored"], "mic_log2"].eq(8).all():
        raise ValueError("Right-censored targets must use log2 boundary 8")
    features = pd.read_csv(require_file(args.peptide_features))
    pc_columns = feature_columns(features)
    scaler = StateFeatureScaler.load(require_file(args.scaler))
    if len(pc_columns) != 32: raise ValueError("V6 requires 32 physicochemical features")
    if scaler.columns != pc_columns:
        raise ValueError("MIC features do not match the frozen state scaler")
    features.loc[:, pc_columns] = scaler.transform(features[pc_columns].to_numpy(dtype=np.float32))
    store = EmbeddingStore.load(args.peptide_embeddings, args.peptide_embedding_metadata)
    if store.global_embeddings.shape[1] != 1536: raise ValueError("V6 requires 1536-dimensional ESM3 embeddings")
    means = pd.read_csv(require_file(args.expert_means))
    mean_cols = ["mu_prior", "mu_mem", "mu_int"]
    if not set(mean_cols).issubset(means.columns):
        raise ValueError(f"Expert means missing columns: {mean_cols}")
    if not np.isfinite(means[mean_cols].to_numpy(float)).all() or not means[mean_cols].ge(0).all().all() or not means[mean_cols].le(1).all().all():
        raise ValueError("Invalid frozen expert means")
    key = args.peptide_id_column
    joined = ledger.merge(features[[key] + feature_columns(features)], left_on="peptide_id", right_on=key, validate="one_to_one")
    joined = joined.merge(means[["peptide_id"] + mean_cols], on="peptide_id", validate="one_to_one")
    if len(joined) != len(ledger):
        raise ValueError("Features and expert means must cover every MIC record")
    ledger_record_hashes = {str(row["peptide_id"]): hashlib.sha256(pd.Series(row).to_json().encode()).hexdigest() for row in ledger.to_dict("records")}
    validation_frame = joined[joined["split"] == "validation"]
    joined = joined[joined["split"] == "train"]
    if joined.empty or validation_frame.empty:
        raise ValueError("Nonempty train and reserved validation sets required")
    ids = joined["peptide_id"].astype(str).tolist()
    model = MICRegressionHead(
        peptide_dim=store.global_embeddings.shape[1],
        physchem_dim=len(feature_columns(features)),
    )
    if args.parent_checkpoint:
        parent = torch.load(require_file(args.parent_checkpoint), map_location="cpu", weights_only=True)
        if parent["metadata"].get("state_name") != manifest.parent_state or not parent["metadata"].get("paper_prequential_validated"):
            raise ValueError("Parent checkpoint does not match state manifest")
        if parent["metadata"].get("frozen_artifact_hashes") != {key:input_hashes[key] for key in FROZEN_ARTIFACTS}:
            raise ValueError("Frozen scorer artifacts changed since parent state")
        previous_records = parent["metadata"].get("ledger_record_hashes", {})
        if not previous_records or any(ledger_record_hashes.get(key) != digest for key,digest in previous_records.items()):
            raise ValueError("Previously released ledger records must remain unchanged in cumulative updates")
        old_splits = parent["metadata"].get("ledger_splits", {})
        new_splits = dict(zip(ledger.peptide_id.astype(str), ledger.split))
        if not old_splits or any(new_splits.get(key) != value for key,value in old_splits.items()):
            raise ValueError("Cumulative ledger lost records or changed reserved splits")
        if {k for k,v in old_splits.items() if v == "validation"} != {k for k,v in new_splits.items() if v == "validation"}:
            raise ValueError("Reserved validation membership must remain frozen")
        if pd.Timestamp(parent["metadata"]["data_cutoff"]) > cutoff: raise ValueError("Parent cutoff exceeds current cutoff")
        input_paths["parent_checkpoint"] = args.parent_checkpoint
        input_hashes["parent_checkpoint"] = sha256_file(args.parent_checkpoint)
        model.load_state_dict(parent["state_dict"])
    validation = (
        store.global_by_record_id(validation_frame["peptide_id"].astype(str).tolist()),
        validation_frame[pc_columns].to_numpy(dtype=np.float32),
        validation_frame[mean_cols].to_numpy(dtype=np.float32),
        validation_frame["mic_log2"].to_numpy(dtype=np.float32),
        validation_frame["mic_censored"].to_numpy(dtype=bool),
    )
    losses = train_mic_head(
        model,
        store.global_by_record_id(ids),
        joined[feature_columns(features)].to_numpy(dtype=np.float32),
        joined[mean_cols].to_numpy(dtype=np.float32),
        joined["mic_log2"].to_numpy(dtype=np.float32),
        joined["mic_censored"].astype(bool).to_numpy(),
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device,
        validation=validation, weight_decay=args.weight_decay, delta=args.huber_delta, patience=args.patience,
    )
    if any(sha256_file(input_paths[key]) != digest for key,digest in input_hashes.items()):
        raise ValueError("MIC training input changed during fitting")
    save_checkpoint(model, Path(args.output), metadata={
        "target_semantics": "measured_mic", "target_definition": "panel_best_mic", "target_divisor": 1,
        "architecture": "1571->16->1",
        "input_hashes": input_hashes,
        "frozen_artifact_hashes": {key:input_hashes[key] for key in FROZEN_ARTIFACTS},
        "ledger_splits": dict(zip(ledger.peptide_id.astype(str), ledger.split)),
        "ledger_record_hashes": ledger_record_hashes,
        "peptide_dim": store.global_embeddings.shape[1],
        "physchem_dim": len(pc_columns),
        "expert_dim": 3,
        "losses": losses,
        "training_config": vars(args),
        "target": "log2(MIC)",
        "censor_limit_log2": 8.0,
        "paper_prequential_validated": True,
        "state_name": manifest.state_name,
        "visible_round": manifest.visible_round,
        "current_round": args.current_round,
        "data_cutoff": manifest.data_cutoff,
        "scaler": str(Path(args.scaler)),
    })
    print(f"MIC head checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
