"""Pinned, verified first-run downloads; complete installations work offline."""
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from filelock import FileLock

MANIFEST = Path(__file__).resolve().parents[1] / 'configs/assets/downloads.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(repo, revision, name):
    from huggingface_hub import hf_hub_download
    for attempt in range(3):
        try:
            return Path(hf_hub_download(repo_id=repo, revision=revision, filename=name))
        except Exception as exc:
            if attempt == 2:
                raise ValueError(
                    f'Cannot download {repo}/{name}. Check internet access or HTTPS_PROXY; '
                    'retry the same command to resume. For offline use, copy a complete assets directory.'
                ) from exc
            time.sleep(2 ** attempt)


def ensure_assets(root, paths=None):
    """Never replace existing model pointers or custom calibrated bundles."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    requested = None if paths is None else {Path(p).resolve() for p in paths}
    esm_root = (root / "shared/esm3").resolve()
    need_esm = requested is None or any(p.is_relative_to(esm_root) for p in requested)
    with FileLock(str(root / '.download.lock')):
        for group, destination in [('models', root), ('esm3', root / 'shared/esm3')]:
            spec = manifest[group]
            for name, expected in spec['files'].items():
                target = destination / name
                if requested is not None and target.resolve() not in requested and not (group == 'esm3' and need_esm):
                    continue
                # Calibration promotion changes only the registry pointer.
                mutable = group == 'models' and name == 'models.json'
                if target.is_file():
                    if mutable:
                        continue
                    if target.stat().st_size == expected['size'] and digest(target) == expected['sha256']:
                        continue
                    raise ValueError(f'Asset checksum mismatch: {target}. Move this file aside and retry to download the pinned original.')
                print(f'Downloading {group}: {name}', flush=True)
                cached = fetch(spec['repo_id'], spec['revision'], name)
                if cached.stat().st_size != expected['size'] or digest(cached) != expected['sha256']:
                    raise ValueError(f'Download checksum mismatch: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix='.download-', dir=target.parent)
                os.close(fd)
                try:
                    shutil.copyfile(cached, tmp)
                    os.replace(tmp, target)
                finally:
                    Path(tmp).unlink(missing_ok=True)
