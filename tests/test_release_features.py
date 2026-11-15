import json
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from mdramp.release_features import prepare_public_features
from src.features import build_esm3_embeddings as build
from src.features.esm3_encoder import ESM3Embedding
from src.utils.hash import sha256_file


def fixture(tmp_path, monkeypatch):
    root = tmp_path / 'public'
    root.mkdir()
    source = root / 'peptides_metadata.csv'
    pd.DataFrame({'record_id': ['p1', 'p2'], 'sequence': ['ACD', 'KKLL']}).to_csv(source, index=False)
    weights = tmp_path / 'weights'
    weights.write_bytes(b'fixed-test-weights')
    (root / 'checksums.json').write_text(json.dumps({source.name: sha256_file(source)}))
    plan = {'stages': [{'options': {'peptide-embeddings': 'DATA/peptides.npz',
                                   'peptide-embedding-metadata': 'DATA/peptides_metadata.csv'}}]}
    monkeypatch.setattr('mdramp.panel_embed.chunked_runtime', lambda _: nullcontext())
    monkeypatch.setattr(build, '_runtime_fingerprint', lambda _: {'test': 1})
    return root, source, weights, plan


def test_rebuild_then_reuse_and_detect_corruption(tmp_path, monkeypatch):
    root, source, weights, plan = fixture(tmp_path, monkeypatch)
    calls = []
    class Encoder:
        def __init__(self, **kwargs): pass
        def encode_sequence(self, record_id, sequence):
            calls.append(record_id)
            residue = np.ones((len(sequence), 4), dtype=np.float32)
            return ESM3Embedding(record_id, sequence, residue, residue.mean(0))
    monkeypatch.setattr(build, 'ESM3Encoder', Encoder)
    result = prepare_public_features(plan, root, weights)
    archive = result['stages'][0]['options']['peptide-embeddings']
    assert calls == ['p1', 'p2']
    assert plan['stages'][0]['options']['peptide-embeddings'] == 'DATA/peptides.npz'
    assert prepare_public_features(plan, root, weights) == result
    assert calls == ['p1', 'p2']
    with np.load(archive) as data:
        assert data['global_embeddings'].shape == (2, 4)
    from pathlib import Path
    Path(archive).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare_public_features(plan, root, weights)


def test_interrupted_encoding_resumes_without_reencoding_completed_rows(tmp_path, monkeypatch):
    root, source, weights, plan = fixture(tmp_path, monkeypatch)
    calls = []
    class Encoder:
        def __init__(self, **kwargs): pass
        def encode_sequence(self, record_id, sequence):
            calls.append(record_id)
            if calls == ['p1', 'p2']:
                raise RuntimeError('interrupted')
            residue = np.ones((len(sequence), 4), dtype=np.float32)
            return ESM3Embedding(record_id, sequence, residue, residue.mean(0))
    monkeypatch.setattr(build, 'ESM3Encoder', Encoder)
    with pytest.raises(RuntimeError, match='interrupted'):
        prepare_public_features(plan, root, weights)
    prepare_public_features(plan, root, weights)
    assert calls == ['p1', 'p2', 'p2']


def test_reject_changed_sequence_table(tmp_path, monkeypatch):
    root, source, weights, plan = fixture(tmp_path, monkeypatch)
    source.write_text('record_id,sequence\np1,YYY\n')
    with pytest.raises(ValueError, match='input checksum'):
        prepare_public_features(plan, root, weights)


def test_missing_checksum_manifest_is_rejected(tmp_path, monkeypatch):
    root, source, weights, plan = fixture(tmp_path, monkeypatch)
    (root / 'checksums.json').unlink()
    with pytest.raises(FileNotFoundError, match='checksums missing'):
        prepare_public_features(plan, root, weights)
