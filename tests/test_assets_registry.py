import pytest
from src.assets.cli import register,status,doctor


def test_registry_detects_changes_and_rejects_overwrite(tmp_path):
    asset=tmp_path/'model.bin';asset.write_bytes(b'fixture')
    registry=tmp_path/'registry.json'
    record=register(registry,'teacher-v1',asset,'local fixture','v1')
    assert record['files'][0]['bytes']==7
    assert status(registry)['teacher-v1']['ok']
    with pytest.raises(ValueError):register(registry,'teacher-v1',asset,'local','v2')
    asset.write_bytes(b'changed')
    assert not status(registry)['teacher-v1']['ok']
    asset.unlink()
    assert not status(registry)['teacher-v1']['ok']


def test_empty_assets_rejected(tmp_path):
    folder=tmp_path/'empty';folder.mkdir()
    with pytest.raises(ValueError):register(tmp_path/'registry','empty',folder,'local','fixture')
    assert 'python' in doctor()


def test_malformed_schema_rejected(tmp_path):
    registry=tmp_path/'bad.json';registry.write_text('{"schema_version": 99, "assets": {}}')
    with pytest.raises(ValueError):status(registry)


def test_concurrent_registration_preserves_unique_entries(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    asset=tmp_path/'asset';asset.write_bytes(b'x')
    registry=tmp_path/'registry.json'
    def add(i):return register(registry,f'asset-{i}',asset,'fixture','v1')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(add,range(8)))
    assert len(status(registry))==8
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(register,registry,'same',asset,'fixture','v1') for _ in range(2)]
    assert sum(f.exception() is None for f in futures)==1
