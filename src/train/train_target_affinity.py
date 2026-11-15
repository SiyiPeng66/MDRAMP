"""Train Target Affinity Expert from TPepPro pair-level teacher scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import ProjectedPairAffinityExpert
from src.train.train_utils import require_file, save_checkpoint
from src.utils.hash import hash_existing, sha256_file


def validate_curation(pair_path, manifest_path=None, data_mode="strict-v6"):
    """Validate locally available S6.2 screening evidence, never infer it from labels."""
    if data_mode == "public-benchmark":
        if manifest_path is not None:
            raise ValueError("Benchmark mode must not claim a strict curation manifest")
        return {"data_mode":data_mode,"v6_pair_curation_complete":False,
                "teacher_screening":{"status":"not_performed"},"curation_manifest_sha256":None}, {}
    if data_mode != "strict-v6" or manifest_path is None:
        raise ValueError("Strict V6 interaction training requires --curation-manifest")
    manifest_path=Path(manifest_path).resolve()
    manifest_hash=sha256_file(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    screening=manifest.get("teacher_screening",{})
    if (manifest.get("schema_version") != 1 or manifest.get("v6_pair_curation_complete") is not True
            or screening.get("status") != "completed"
            or screening.get("high_confidence_threshold") != 0.8):
        raise ValueError("Incomplete V6 TPepPro screening or incorrect high-confidence threshold")
    pair_path=Path(pair_path).resolve()
    outputs=manifest.get("output_sha256",{})
    matches=[value for name,value in outputs.items()
             if (manifest_path.parent/name).resolve() == pair_path]
    if len(matches)!=1 or sha256_file(pair_path)!=matches[0]:
        raise ValueError("Curation manifest does not bind the selected pair CSV")
    evidence={str(manifest_path):manifest_hash,str(pair_path):matches[0]}
    for name in ("scores","weights"):
        artifact=screening.get(name,{})
        if not isinstance(artifact,dict) or not artifact.get("path") or not artifact.get("sha256"):
            raise ValueError(f"Missing TPepPro {name} provenance")
        path=(manifest_path.parent/artifact["path"]).resolve()
        if not path.is_file() or sha256_file(path)!=artifact["sha256"]:
            raise ValueError(f"TPepPro {name} artifact missing or hash mismatch")
        evidence[str(path)]=artifact["sha256"]
    # Verify the builder's actual exclusion, not merely its completion declaration.
    pairs=pd.read_csv(pair_path,dtype={"pair_id":str})
    if not {"pair_id","label_type"}.issubset(pairs):
        raise ValueError("Strict V6 pair CSV requires pair_id and label_type")
    if pairs.pair_id.isna().any() or pairs.pair_id.duplicated().any():
        raise ValueError("Strict V6 pair IDs must be nonmissing and unique")
    score_path=(manifest_path.parent/screening["scores"]["path"]).resolve()
    scores=pd.read_csv(score_path,dtype={"pair_id":str})
    if not {"pair_id","s_aff_pair"}.issubset(scores):
        raise ValueError("Teacher scores require pair_id and s_aff_pair")
    if scores.pair_id.isna().any() or scores.pair_id.duplicated().any():
        raise ValueError("Teacher score pair IDs must be nonmissing and unique")
    background=pairs.loc[pairs.label_type.eq("background"),"pair_id"]
    lookup=scores.set_index("pair_id")
    if not set(background).issubset(lookup.index):
        raise ValueError("Missing teacher scores for operational background pairs")
    values=pd.to_numeric(lookup.loc[background,"s_aff_pair"],errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(values)&(values>=0)&(values<0.8)).all():
        raise ValueError("Background teacher scores must be finite in [0,0.8); high-confidence positives excluded")
    verify_curation_evidence(evidence)
    if sha256_file(manifest_path)!=manifest_hash:
        raise ValueError("Curation manifest changed during validation")
    return {"data_mode":data_mode,"v6_pair_curation_complete":True,
            "teacher_screening":screening,"curation_manifest_sha256":manifest_hash},evidence


def verify_curation_evidence(evidence):
    if any(not Path(path).is_file() or sha256_file(path)!=value for path,value in evidence.items()):
        raise ValueError("Curation evidence changed during training; checkpoint not published")


def validate_training_partition(pair_scores: pd.DataFrame) -> None:
    if 'split' in pair_scores and not pair_scores['split'].eq('train').all():
        raise ValueError('Interaction training accepts only train pair rows, never validation/test')


def build_pair_arrays(
    pair_scores: pd.DataFrame,
    peptide_features: pd.DataFrame,
    peptide_store: EmbeddingStore,
    target_store: EmbeddingStore,
    *,
    peptide_id_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pc_cols = feature_columns(peptide_features)
    peptide_lookup = peptide_features.set_index(peptide_id_column)
    peptide_embeddings = []
    target_embeddings = []
    physchem = []
    labels = []
    for row in pair_scores.itertuples(index=False):
        explicit = getattr(row, "explicit_label", float("nan"))
        teacher = float(getattr(row, "s_aff_pair", float("nan")))
        if pd.notna(explicit):
            label = float(explicit)
            if label not in (0., 1.): raise ValueError("Explicit pair labels must be 0 or 1")
        elif getattr(row, "label_type", "") == "background": label = 0.
        elif np.isfinite(teacher) and 0 <= teacher <= 1 and (teacher <= .2 or teacher >= .8): label = teacher
        else: continue
        peptide_id = str(row.peptide_id)
        target_id = str(row.target_id)
        peptide_embeddings.append(peptide_store.global_by_record_id([peptide_id])[0])
        target_embeddings.append(target_store.global_by_record_id([target_id])[0])
        physchem.append(peptide_lookup.loc[peptide_id, pc_cols].to_numpy(dtype=np.float32))
        labels.append(label)
    if not labels: raise ValueError("No valid pair supervision")
    return (
        np.asarray(peptide_embeddings, dtype=np.float32),
        np.asarray(target_embeddings, dtype=np.float32),
        np.asarray(physchem, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def train_pair_model(
    model: ExpertEnsemble,
    peptide_x: np.ndarray,
    target_x: np.ndarray,
    physchem_x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    weight_decay: float = 0.01,
) -> list[list[float]]:
    loss_fn = torch.nn.BCELoss()
    pep = torch.tensor(peptide_x, dtype=torch.float32)
    tar = torch.tensor(target_x, dtype=torch.float32)
    pc = torch.tensor(physchem_x, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.float32)
    histories = []
    for seed, member in zip(model.seeds, model.members):
        print(f'interaction member seed={seed} samples={len(y)} epochs={epochs}', flush=True)
        torch.manual_seed(seed)
        member.to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=learning_rate, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        losses = []
        for epoch in range(epochs):
            batch_losses = []
            for idx in torch.randperm(len(yy), generator=generator).split(batch_size):
                pred = member(pep[idx].to(device), tar[idx].to(device), pc[idx].to(device))
                loss = loss_fn(pred, yy[idx].to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite interaction training loss')
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(batch_losses)))
            if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                print(f'interaction seed={seed} epoch={epoch+1}/{epochs} loss={losses[-1]:.6f}', flush=True)
        histories.append(losses)
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-scores", required=True)
    parser.add_argument("--data-mode", choices=("strict-v6","public-benchmark"), default="strict-v6")
    parser.add_argument("--curation-manifest", help="Required in strict-v6 mode: S6.2 teacher screening with artifact hashes")
    parser.add_argument("--peptide-features", required=True)
    parser.add_argument("--peptide-embeddings", required=True)
    parser.add_argument("--peptide-embedding-metadata", required=True)
    parser.add_argument("--target-embeddings", required=True)
    parser.add_argument("--target-embedding-metadata", required=True)
    parser.add_argument("--peptide-id-column", default="sequence")
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--output", default="outputs/checkpoints/M_R0/target_affinity_expert.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ensemble-seeds", type=int, nargs=5, default=[11,23,37,51,73])
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid optimization hyperparameters")
    if len(set(args.ensemble_seeds)) != 5: parser.error("Five independent seeds required")

    input_paths = [args.pair_scores, args.peptide_features, args.peptide_embeddings,
                   args.peptide_embedding_metadata, args.target_embeddings,
                   args.target_embedding_metadata, args.scaler]
    input_hashes = hash_existing(input_paths)
    curation, curation_evidence = validate_curation(args.pair_scores, args.curation_manifest, args.data_mode)
    pair_scores = pd.read_csv(require_file(args.pair_scores))
    validate_training_partition(pair_scores)
    if "label_type" not in pair_scores:
        raise ValueError("Paper training pairs require label_type=positive/background")
    label_type = pair_scores["label_type"].astype(str)
    unexpected = sorted(set(label_type) - {"positive", "background"})
    if unexpected:
        raise ValueError(f"Unexpected pair label types: {unexpected}")
    counts = label_type.value_counts()
    if counts.get("positive", 0) != counts.get("background", 0):
        raise ValueError("Paper interaction training requires 1:1 positive/background pairs")
    peptide_features = pd.read_csv(require_file(args.peptide_features))
    pc_columns = feature_columns(peptide_features)
    scaler = StateFeatureScaler.load(require_file(args.scaler))
    if scaler.columns != pc_columns:
        raise ValueError("Interaction features do not match the frozen scaler")
    peptide_features.loc[:, pc_columns] = scaler.transform(
        peptide_features[pc_columns].to_numpy(dtype=np.float32)
    )
    peptide_store = EmbeddingStore.load(args.peptide_embeddings, args.peptide_embedding_metadata)
    target_store = EmbeddingStore.load(args.target_embeddings, args.target_embedding_metadata)
    peptide_x, target_x, physchem_x, y = build_pair_arrays(
        pair_scores,
        peptide_features,
        peptide_store,
        target_store,
        peptide_id_column=args.peptide_id_column,
    )
    peptide_dim = peptide_store.global_embeddings.shape[1]
    target_dim = target_store.global_embeddings.shape[1]
    physchem_dim = len(feature_columns(peptide_features))
    seeds = tuple(args.ensemble_seeds)
    model = ExpertEnsemble(
        lambda: ProjectedPairAffinityExpert(peptide_dim, target_dim, physchem_dim),
        seeds=seeds,
    )
    losses = train_pair_model(
        model,
        peptide_x,
        target_x,
        physchem_x,
        y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device, weight_decay=args.weight_decay,
    )
    if hash_existing(input_paths) != input_hashes:
        raise ValueError('Interaction input changed during training; checkpoint not published')
    verify_curation_evidence(curation_evidence)
    save_checkpoint(
        model,
        Path(args.output),
        metadata={
            **curation,
            "training_config": vars(args),
            "input_hashes": input_hashes,
            "peptide_dim": peptide_dim,
            "target_dim": target_dim,
            "physchem_dim": physchem_dim,
            "losses": losses,
            "source": "TPepPro",
            "architecture": "ProjectedPairAffinityExpert",
            "projection_dim": 256,
            "bilinear_rank": 64,
            "seeds": list(seeds),
            "pair_sampling": "1:1 positive/operational_background",
            "scaler": str(Path(args.scaler)),
        },
    )
    print(f"target affinity checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
