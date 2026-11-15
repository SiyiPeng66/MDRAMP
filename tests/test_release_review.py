import json
from pathlib import Path
import pytest
from mdramp import release_assets

def test_promotion_rejects_foundation_without_head(tmp_path,monkeypatch):
    registry=tmp_path/'models.json';registry.write_text(json.dumps({'FinalModel':'original.json'}))
    monkeypatch.setattr(release_assets,'ASSETS',tmp_path)
    monkeypatch.setattr(release_assets,'resolve_model',lambda _: {'checkpoint':None})
    with pytest.raises(ValueError,match='calibrated MIC'):release_assets.promote_final(tmp_path/'scorer.json')
    assert json.loads(registry.read_text())['FinalModel']=='original.json'

def test_custom_checkpoint_rejects_companion_pointing_elsewhere(tmp_path):
    (tmp_path/'scorer.json').write_text(json.dumps({'mic_checkpoint':'other.pt'}))
    with pytest.raises(ValueError,match='does not match'):release_assets.resolve_model(str(tmp_path/'model.pt'))

def test_relative_generator_resolves_with_bundle(tmp_path):
    (tmp_path/'scorer.json').write_text(json.dumps({'generator':'../shared/generator.pt'}))
    b=release_assets.resolve_model(str(tmp_path/'scorer.json'))
    assert Path(b['generator']).resolve()==(tmp_path.parent/'shared/generator.pt').resolve()
