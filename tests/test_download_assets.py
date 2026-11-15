import hashlib
import json
from pathlib import Path
import pytest
from mdramp import download_assets as d


def fixture(monkeypatch, tmp_path):
    payload=tmp_path/'payload';payload.write_bytes(b'weights')
    info={'size':7,'sha256':hashlib.sha256(b'weights').hexdigest()}
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'models':{'repo_id':'test/model','revision':'pinned','files':{'FinalModel/model.pt':info}},'esm3':{'repo_id':'test/esm','revision':'pinned','files':{'data/weights/esm3_sm_open_v1.pth':info,'data/tokenizer.txt':info}}}))
    monkeypatch.setattr(d,'MANIFEST',manifest)
    calls=[]
    def fetch(repo,revision,name):calls.append((repo,revision,name));return payload
    monkeypatch.setattr(d,'fetch',fetch)
    return calls


def test_download_verify_reuse_and_corruption(monkeypatch,tmp_path):
    calls=fixture(monkeypatch,tmp_path);root=tmp_path/'assets'
    d.ensure_assets(root);assert len(calls)==3
    d.ensure_assets(root);assert len(calls)==3
    (root/'FinalModel/model.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):d.ensure_assets(root)


def test_custom_offline_and_esm_companions(monkeypatch,tmp_path):
    calls=fixture(monkeypatch,tmp_path);root=tmp_path/'assets'
    d.ensure_assets(root,[tmp_path/'custom/model.pt']);assert not calls
    d.ensure_assets(root,[root/'shared/esm3/data/weights/esm3_sm_open_v1.pth'])
    assert len(calls)==2
    assert not (root/'FinalModel/model.pt').exists()


def test_bad_download_not_installed(monkeypatch,tmp_path):
    fixture(monkeypatch,tmp_path);(tmp_path/'payload').write_bytes(b'invalid')
    with pytest.raises(ValueError,match='checksum'):d.ensure_assets(tmp_path/'assets')
    assert not (tmp_path/'assets/FinalModel/model.pt').exists()


def test_resolve_self_contained_custom_offline(monkeypatch, tmp_path):
    from mdramp import release_assets
    calls = fixture(monkeypatch, tmp_path)
    root = tmp_path / 'assets'
    custom = tmp_path / 'custom'
    custom.mkdir()
    (custom / 'model.pt').write_bytes(b'custom checkpoint')
    (custom / 'generator.pt').write_bytes(b'custom generator')
    (custom / 'scorer.json').write_text(json.dumps({'mic_checkpoint': 'model.pt'}))
    monkeypatch.setattr(release_assets, 'ASSETS', root)
    bundle = release_assets.resolve_model(str(custom / 'model.pt'))
    assert bundle['checkpoint'] == str(custom / 'model.pt')
    assert not calls
    assert not (root / 'FinalModel/model.pt').exists()


def test_promoted_pointer_preserved_without_original_model_download(monkeypatch, tmp_path):
    from mdramp import release_assets
    calls = fixture(monkeypatch, tmp_path)
    root = tmp_path / 'assets'
    root.mkdir()
    custom = tmp_path / 'calibrated'
    custom.mkdir()
    (custom / 'model.pt').write_bytes(b'new checkpoint')
    (custom / 'scorer.json').write_text(json.dumps({'mic_checkpoint': 'model.pt'}))
    registry = {'FinalModel': '../calibrated/scorer.json', 'BaseModel': 'BaseModel/scorer.json'}
    (root / 'models.json').write_text(json.dumps(registry))
    monkeypatch.setattr(release_assets, 'ASSETS', root)
    bundle = release_assets.resolve_model('FinalModel')
    assert Path(bundle['checkpoint']).resolve() == custom / 'model.pt'
    assert json.loads((root / 'models.json').read_text()) == registry
    assert not calls
    assert not (root / 'FinalModel/model.pt').exists()
