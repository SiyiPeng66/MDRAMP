"""Resumable, provenance-bound inference with the actual PF teacher."""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd

from src.utils.hash import sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def model_identity(root):
    files = [root / 'model.pkl', *sorted((root / 'prot_bert_bfd').rglob('*'))]
    return {str(p.relative_to(root)): sha256_file(p) for p in files if p.is_file()}


def score_resumable(input_csv, output_csv, *, model_root, sequence_column, id_column, device):
    from src.external.pore_forming_wrapper import PoreFormingTeacher
    input_csv, output_csv, model_root = Path(input_csv), Path(output_csv), Path(model_root)
    input_hash = sha256_file(input_csv)
    frame = pd.read_csv(input_csv, dtype={id_column: str, sequence_column: str})
    if frame.empty or frame[[id_column, sequence_column]].isna().any().any() or frame[id_column].duplicated().any():
        raise ValueError('PF input must have nonempty unique IDs and sequences')
    if sha256_file(input_csv) != input_hash: raise ValueError('PF input changed while reading')
    binding = {'schema_version': 1, 'input_sha256': input_hash, 'device': device,
               'id_column': id_column, 'sequence_column': sequence_column,
               'model': model_identity(model_root),
               'implementation': {name: sha256_file(Path(__file__).with_name(name))
                                  for name in ('pore_cache.py', 'pore_runtime.py', 'pore_forming_wrapper.py')},
               'versions': {name: importlib.metadata.version(name)
                            for name in ('torch', 'transformers', 'scikit-learn', 'numpy')}}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache = output_csv.parent / ('_cache_' + output_csv.stem)
    cache.mkdir(exist_ok=True)
    with (cache / '.lock').open('a+') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('PF cache is already in use') from error
        manifest_path = cache / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('binding') != binding: raise ValueError('PF input/model/runtime changed; use a new output')
        else:
            if output_csv.exists() or any(p.name != '.lock' for p in cache.iterdir()):
                raise ValueError('Existing PF output/cache has no matching manifest')
            manifest = {'binding': binding, 'completed_records': 0, 'record_count': len(frame)}
        completed = manifest['completed_records']
        manifest.update(status='running'); manifest.pop('error', None)
        atomic_json(manifest_path, manifest)
        teacher, rows = None, []
        try:
            for index, record in enumerate(frame.to_dict('records')):
                identifier, sequence = record[id_column], record[sequence_column]
                key = digest([identifier, sequence, binding])
                path = cache / (key + '.json')
                if index < completed and not path.exists(): raise ValueError('Completed PF cache entry missing')
                if path.exists():
                    cached = json.loads(path.read_text()); row = cached['row']
                    if cached.get('sha256') != digest(row) or row['record_id'] != identifier or row['sequence'] != sequence:
                        raise ValueError('PF cached record changed')
                else:
                    if teacher is None: teacher = PoreFormingTeacher(model_root, device)
                    row = {'record_id': identifier, 'sequence': sequence,
                           'pore_teacher_score': teacher.score(sequence), 'teacher_model': 'Pore-Forming'}
                    atomic_json(path, {'row': row, 'sha256': digest(row)})
                value = float(row['pore_teacher_score'])
                if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError('Invalid PF teacher probability')
                rows.append(row); manifest['completed_records'] = index + 1
                atomic_json(manifest_path, manifest)
                if (index + 1) % 25 == 0 or index + 1 == len(frame):
                    print(f'PF teacher: {index+1}/{len(frame)}', flush=True)
            if sha256_file(input_csv) != input_hash or model_identity(model_root) != binding['model']:
                raise ValueError('PF input/model changed during inference')
            if any(sha256_file(Path(__file__).with_name(name)) != value for name, value in binding['implementation'].items()):
                raise ValueError('PF implementation changed during inference')
            result = pd.DataFrame(rows)
            fd, temporary = tempfile.mkstemp(dir=output_csv.parent, prefix='.' + output_csv.name)
            try:
                with os.fdopen(fd, 'w') as handle:
                    result.to_csv(handle, index=False); handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, output_csv)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
            manifest.update(status='complete', output_sha256=sha256_file(output_csv))
            atomic_json(manifest_path, manifest)
            return result
        except BaseException as error:
            manifest.update(status='failed', error=str(error)); atomic_json(manifest_path, manifest)
            raise
