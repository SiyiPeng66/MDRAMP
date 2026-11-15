"""Organize existing sequence inputs without rebuilding or validating historical splits.

This is a data inventory/replay preparation step, NOT a training-ready dataset
builder. It preserves historical split assignments exactly by sequence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.data.parse_public import parse_public_rawdata, clean_and_deduplicate
from src.data.parse_targets import parse_targets
from src.data.parse_private_round import parse_private_round
from src.utils.hash import sha256_file


def peptide_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()[:16]


def preserve_split(deduplicated: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    required = {'sequence', 'cluster_id', 'split'}
    if not required.issubset(historical):
        raise ValueError(f'Historical split missing columns: {sorted(required - set(historical))}')
    if historical[list(required)].isna().any().any():
        raise ValueError('Historical sequence/cluster/split values must not be missing')
    if any(historical[column].astype(str).str.strip().eq('').any() for column in required):
        raise ValueError('Historical sequence/cluster/split values must not be empty')
    if historical.sequence.duplicated().any() or deduplicated.sequence.duplicated().any():
        raise ValueError('Historical and newly deduplicated sequences must each be unique')
    if not set(historical.split).issubset({'train', 'validation', 'test'}):
        raise ValueError('Historical split values must be train, validation or test')
    if (historical.groupby('cluster_id').split.nunique() != 1).any():
        raise ValueError('Historical cluster spans multiple splits')
    current, old = set(deduplicated.sequence), set(historical.sequence)
    if current != old:
        raise ValueError(f'Historical sequence set mismatch: new_only={len(current-old)}, historical_only={len(old-current)}')
    result = deduplicated.merge(historical[['sequence', 'cluster_id', 'split']], on='sequence', validate='one_to_one')
    result.insert(0, 'peptide_id', result.sequence.map(peptide_id))
    if result.peptide_id.duplicated().any():
        raise ValueError('Truncated peptide identifier collision')
    return result.sort_values(['split', 'cluster_id', 'sequence']).reset_index(drop=True)


def _inventory(root: Path, include_private: bool = False) -> dict[str, str]:
    raw = root / 'data/rawdata'
    if not raw.is_dir():
        raise FileNotFoundError(f'Missing public source directory: {raw}')
    paths = sorted(p for p in raw.rglob('*') if p.is_file())
    paths += [root/'outputs/processed/public_cluster_split.csv', root/'data/target/target.fa']
    if include_private:
        paths.append(root/'data/private/R0.txt')
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def _length_audit(public: pd.DataFrame) -> dict:
    rows = {}
    for name, frame in public.groupby('split', sort=True):
        lengths = frame.sequence.str.len()
        rows[name] = {'count': len(frame), 'clusters': int(frame.cluster_id.nunique()),
                      'minimum': int(lengths.min()), 'median': float(lengths.median()),
                      'mean': float(lengths.mean()), 'maximum': int(lengths.max()),
                      'length_histogram': {str(k): int(v) for k, v in lengths.value_counts().sort_index().items()}}
    return {'by_split': rows, 'interpretation': 'Descriptive length distributions only; historical assignments are unchanged. No randomization, homology isolation or lack of length bias is certified.'}


def prepare(root: str | Path, output_dir: str | Path, include_private: bool = False) -> dict:
    root, destination = Path(root).resolve(), Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing existing output directory: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ('.' + destination.name + '.prepare.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary = None
    try:
        os.close(fd)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Refusing existing output directory: {destination}')
        inputs = _inventory(root, include_private)
        raw = parse_public_rawdata(root/'data/rawdata')
        if raw.empty:
            raise ValueError('No public sequence records parsed')
        cleaned, deduplicated = clean_and_deduplicate(raw)
        if deduplicated.empty:
            raise ValueError('No eligible public sequences remain')
        history = pd.read_csv(root/'outputs/processed/public_cluster_split.csv', dtype={'sequence': str, 'cluster_id': str, 'split': str})
        public = preserve_split(deduplicated, history)
        temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
        public.to_csv(temporary/'public_sequences.csv', index=False)
        # Every raw record is retained, including invalid records and repeated sources.
        records = cleaned.copy()
        records.insert(0, 'source_record_index', range(1, len(records) + 1))
        records['source_file'] = records.source_file.map(lambda p: str(Path(p).relative_to(root)))
        records = records.merge(public[['sequence', 'peptide_id', 'cluster_id', 'split']], on='sequence', how='left', validate='many_to_one')
        if len(records) != len(raw):
            raise ValueError('Source-record preservation failed')
        records.to_csv(temporary/'public_records.csv', index=False)
        targets = parse_targets(root/'data/target/target.fa', temporary/'target_corpus.csv')
        private, private_sequences = None, None
        if include_private:
            private = parse_private_round(root/'data/private/R0.txt', temporary/'R0_records.csv', round_name='R0')
            private_sequences = private.loc[private.is_valid, ['private_record_id', 'round', 'sequence', 'length', 'data_mode']].copy()
            private_sequences.insert(0, 'peptide_id', private_sequences.sequence.map(peptide_id))
            private_sequences.to_csv(temporary/'R0_sequences.csv', index=False)
        counts = {'public': {'raw_records': len(raw), 'valid_records': int(cleaned.is_valid.sum()),
                             'deduplicated_sequences': len(public), 'clusters': int(public.cluster_id.nunique()),
                             'split_sequences': {str(k): int(v) for k, v in public.split.value_counts().items()},
                             'raw_records_by_source': {str(k): int(v) for k, v in raw.source.value_counts().items()}},
                  'targets': {'raw_records': len(targets), 'valid_records': int(targets.is_valid.sum()),
                              'unique_valid_sequences': int(targets.loc[targets.is_valid, 'target_sequence'].nunique())},
                  'scope': 'public_and_private' if include_private else 'public_only'}
        if include_private:
            counts['R0'] = {'raw_records': len(private), 'valid_records': len(private_sequences), 'unique_valid_sequences': int(private_sequences.sequence.nunique()), 'data_mode': 'sequence_only'}
        if _inventory(root, include_private) != inputs:
            raise RuntimeError('Input inventory changed during preparation; refusing publication')
        outputs = {p.name: sha256_file(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'operation': 'sequence_inventory_preserving_historical_assignments',
                    'root': str(root), 'training_ready': False,
                    'historical_split_unverified': True, 'no_final_target_panel': True,
                    'notes': ['Historical sequence-set equality and within-cluster split consistency checked; historical clustering itself was not rerun or validated.',
                              'target_corpus.csv is a corpus inventory, not the fixed final protein panel.',
                              ('R0_sequences.csv contains only sequence prediction inputs; no MIC, phenotype or training supervision is inferred.' if include_private else 'Private R0–R5 inputs were not read or prepared.'),
                              'Raw public source records are retained without truncating per-record source identifiers.'],
                    'input_sha256': inputs, 'output_sha256': outputs, 'counts': counts,
                    'length_bias_audit': _length_audit(public),
                    'ignored_raw_files': [p for p in inputs if p.startswith('data/rawdata/') and Path(p).suffix.lower() not in {'.fa', '.fasta', '.faa', '.txt', '.csv'}]}
        (temporary/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Output appeared during preparation: {destination}')
        os.rename(temporary, destination)
        temporary = None
        return manifest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Repository root containing existing raw and processed data')
    parser.add_argument('--output-dir', required=True, help='New output directory; existing destinations are rejected')
    parser.add_argument('--include-private', action='store_true', help='Explicitly opt into R0 sequence inventory; off for public-only preparation')
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output_dir, args.include_private), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
