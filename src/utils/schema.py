"""Schema validation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class SchemaError(ValueError):
    pass


def require_columns(df: pd.DataFrame, required: list[str], *, name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SchemaError(f"{name} missing required columns: {missing}")


def validate_csv(path: str | Path, required: list[str], *, name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, required, name=name)
    return df


PUBLIC_SCHEMA = [
    "sequence",
    "length",
    "sources",
    "source_record_ids",
    "source_count",
    "label_amp_prior",
    "cluster_id",
    "split",
]
PRIVATE_SCHEMA = [
    "round",
    "private_record_id",
    "raw_sequence",
    "sequence",
    "length",
    "is_valid",
    "clean_reason",
    "data_mode",
]
TARGET_SCHEMA = [
    "target_id",
    "description",
    "target_sequence",
    "length",
    "is_valid",
    "clean_reason",
]
ESM3_METADATA_SCHEMA = ["record_id", "sequence", "length", "embedding_key", "global_index"]
PORE_SCHEMA = ["record_id", "sequence", "pore_teacher_score", "teacher_model"]
TPEPPRO_PAIR_SCHEMA = ["peptide_id", "target_id", "s_aff_pair"]
TPEPPRO_AGG_SCHEMA = [
    "peptide_id",
    "S_aff",
    "top_target_id",
    "top_target_score",
    "target_count",
    "teacher_model",
]
SCORE_SCHEMA = [
    "rank",
    "sequence",
    "S_prior",
    "S_mem",
    "S_aff",
    "g_prior",
    "E_comp",
    "E_syn",
    "E_mech",
    "p_cons",
    "predicted_log2_mic",
    "A_t",
    "R_t",
]

MIC_LEDGER_SCHEMA = [
    "peptide_id",
    "round",
    "mic_ug_ml",
    "mic_log2",
    "mic_censored",
    "mic_censor_limit_log2",
]

STATE_MANIFEST_SCHEMA = [
    "state_name",
    "parent_state",
    "visible_round",
    "data_cutoff",
    "frozen_components",
    "input_hashes",
]
