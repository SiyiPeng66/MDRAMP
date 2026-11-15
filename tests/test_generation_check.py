import json
import sys
import pytest
from mdramp.generation_check import main


def test_readiness_reports_validation_without_inference(tmp_path, monkeypatch):
    class Scorer:
        def __init__(self, path): assert path == 'scorer.json'
        def validate_artifacts(self): return {'state_name': 'M_R0', 'output_dim': 1}
        def _initialize(self): raise AssertionError('Preflight must not load ESM')
    monkeypatch.setattr('src.score.frozen_scorer.FrozenSequenceScorer', Scorer)
    out=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['check','--scorer-config','scorer.json','--output',str(out)])
    main()
    assert json.loads(out.read_text())['ready'] is True
    with pytest.raises(FileExistsError): main()


def test_readiness_refuses_missing_artifacts(tmp_path, monkeypatch):
    out=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['check','--scorer-config',str(tmp_path/'missing.json'),'--output',str(out)])
    with pytest.raises(SystemExit) as caught: main()
    assert caught.value.code==2
    assert json.loads(out.read_text())['ready'] is False
