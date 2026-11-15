import json
import pytest
from src.external.pore_cache import score_resumable
from src.external import pore_forming_wrapper


def test_pore_cache_recovers_and_rejects_changed_input(tmp_path, monkeypatch):
    source = tmp_path / 'input.csv'
    source.write_text('id,sequence\na,ACDEFGHIKL\nb,GLFDIVKKVV\n')
    root = tmp_path / 'model'; root.mkdir(); (root / 'model.pkl').write_bytes(b'test')
    output = tmp_path / 'scores.csv'; calls = []
    class Teacher:
        def __init__(self, *args): pass
        def score(self, sequence):
            calls.append(sequence)
            if len(calls) == 2: raise RuntimeError('interrupted')
            return 0.9
    monkeypatch.setattr(pore_forming_wrapper, 'PoreFormingTeacher', Teacher)
    kwargs = dict(model_root=root, sequence_column='sequence', id_column='id', device='cpu')
    with pytest.raises(RuntimeError, match='interrupted'):
        score_resumable(source, output, **kwargs)
    assert not output.exists()
    result = score_resumable(source, output, **kwargs)
    assert len(result) == 2 and calls.count('ACDEFGHIKL') == 1
    source.write_text(source.read_text() + 'c,ACDEFGHIKM\n')
    with pytest.raises(ValueError, match='changed'):
        score_resumable(source, output, **kwargs)


def test_pore_cache_rejects_corrupt_score(tmp_path, monkeypatch):
    source = tmp_path / 'input.csv'; source.write_text('id,sequence\na,ACDEFGHIKL\n')
    root = tmp_path / 'model'; root.mkdir(); (root / 'model.pkl').write_bytes(b'test')
    class Teacher:
        def __init__(self, *args): pass
        def score(self, sequence): return 0.9
    monkeypatch.setattr(pore_forming_wrapper, 'PoreFormingTeacher', Teacher)
    output = tmp_path / 'scores.csv'
    kwargs = dict(model_root=root, sequence_column='sequence', id_column='id', device='cpu')
    score_resumable(source, output, **kwargs)
    entry = next(p for p in (tmp_path/'_cache_scores').glob('*.json') if p.name != 'manifest.json')
    record = json.loads(entry.read_text()); record['row']['pore_teacher_score'] = 0.1
    entry.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='changed'):
        score_resumable(source, output, **kwargs)
