"""Rebuild optional public ESM3 caches from versioned sequence metadata.

Distributed sequence/teacher tables are authoritative; computed embeddings are
local, resumable artifacts and never replace the distributed metadata tables.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from src.utils.hash import sha256_file


def _data_path(root, value):
    if not isinstance(value, str) or not value.startswith('DATA/'):
        raise ValueError(f'Expected packaged public data path, got {value!r}')
    path = (root / value[5:]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Public data path escapes its directory')
    return path


def prepare_public_features(plan, public_root, weights_path):
    """Return a plan pointing at verified shipped or recomputed embeddings."""
    root = Path(public_root)
    result = copy.deepcopy(plan)
    checksums_path = root / 'checksums.json'
    if not checksums_path.is_file():
        raise FileNotFoundError(f'Public data checksums missing: {checksums_path}')
    checksums = json.loads(checksums_path.read_text())
    for name, expected in checksums.items():
        path = _data_path(root, 'DATA/' + name)
        if path.suffix == '.npz':
            continue  # Optional large caches are checked below when present.
        if not path.is_file():
            raise FileNotFoundError(f'Public training input missing: {path}')
        if sha256_file(path) != expected:
            raise ValueError(f'Public training input checksum mismatch: {path}')
    requests = []
    for stage in result['stages']:
        options = stage['options']
        for key, value in list(options.items()):
            if not key.endswith('-embeddings'):
                continue
            meta_key = key[:-len('-embeddings')] + '-embedding-metadata'
            source = _data_path(root, options[meta_key])
            archive = _data_path(root, value)
            if not source.is_file():
                raise FileNotFoundError(f'Public sequence metadata missing: {source}')
            expected_source = checksums.get(source.relative_to(root.resolve()).as_posix())
            if not expected_source:
                raise ValueError(f'Public sequence metadata has no checksum: {source}')
            if archive.is_file():
                expected = checksums.get(archive.relative_to(root.resolve()).as_posix())
                if not expected or sha256_file(archive) != expected:
                    raise ValueError(f'Unverified public embedding cache: {archive}; move it aside to rebuild')
                continue
            requests.append((options, key, meta_key, source, archive.stem))
    if not requests:
        return result

    from src.features import build_esm3_embeddings as build
    from mdramp.panel_embed import chunked_runtime
    weights = Path(weights_path)
    weight_hash = sha256_file(weights)
    with chunked_runtime(512):
        runtime = build._runtime_fingerprint(weights)
        for options, key, meta_key, source, prefix in requests:
            identity = {'input_sha256': sha256_file(source), 'checkpoint_sha256': weight_hash,
                        'model_name': 'esm3-sm-open-v1', 'device': 'cpu', 'runtime': runtime}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            directory = root / '.feature_cache' / digest
            receipt = directory / f'{prefix}_metadata.json'
            complete = False
            if receipt.exists():
                info = json.loads(receipt.read_text())
                expected = info.get('output_sha256', {})
                if info.get('status') == 'complete' and expected:
                    complete = all((directory / name).is_file() and sha256_file(directory / name) == checksum
                                   for name, checksum in expected.items())
                    if not complete:
                        raise ValueError(f'Computed embedding cache checksum mismatch: {directory}')
            if not complete:
                print(f'Preparing public ESM3 features: {source.name} (resumes interrupted work)', flush=True)
                build.build_embeddings(dataset=prefix, input_csv=source, output_dir=directory,
                    id_column='record_id', sequence_column='sequence', valid_column=None,
                    model_name='esm3-sm-open-v1', device='cpu', hf_home=root/'.hf_cache',
                    weights_path=str(weights), resume=True)
            options[key] = str(directory / f'{prefix}.npz')
            options[meta_key] = str(directory / f'{prefix}_metadata.csv')
    return result
