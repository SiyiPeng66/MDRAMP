"""Opt-in integration regression using real official RF weights and MERCI."""
import os
from pathlib import Path
import pytest

@pytest.mark.skipif(os.environ.get('MDRAMP_RUN_HEMOPI')!='1',reason='requires official RF artifact and isolated runtime')
def test_official_hybrid_is_independent_of_batch_composition():
    from src.external.hemopi_official import score,score_batch
    root=Path(__file__).resolve().parents[1]
    model=root/'otherModels/hemopi2'; python=root/'.venv-hemopi-rf/bin/python'
    sequences=['KALWQKALWQKALWQKALWQ','GIGAVLKVLTTGLPALISWIKRKRQQ']
    singles=[score(s,model,python) for s in sequences]
    batch=score_batch(sequences,model,python)
    assert singles==batch
    assert all(0<=v<=1 for v in batch)
