"""Build ESM3 embeddings from processed peptide or target CSV files."""

from __future__ import annotations

import argparse
import os
import hashlib
import platform
from importlib import metadata as package_metadata
import json
import tempfile
import zipfile
from contextlib import contextmanager
import numpy as np
from src.utils.hash import sha256_file
from pathlib import Path

import pandas as pd

from src.features.esm3_encoder import ESM3Encoder, ESM3SetupError, save_embeddings_npz


DATASET_CONFIG = {
    "public": {
        "input": "outputs/processed/public_cluster_split.csv",
        "id_column": "sequence",
        "sequence_column": "sequence",
        "valid_column": None,
        "prefix": "public_peptides",
    },
    "private_R0": {
        "input": "outputs/processed/private_R0.csv",
        "id_column": "private_record_id",
        "sequence_column": "sequence",
        "valid_column": "is_valid",
        "prefix": "private_R0_peptides",
    },
    "targets": {
        "input": "outputs/processed/targets.csv",
        "id_column": "target_id",
        "sequence_column": "target_sequence",
        "valid_column": "is_valid",
        "prefix": "targets",
    },
}


def load_records(
    input_csv: Path,
    *,
    id_column: str,
    sequence_column: str,
    valid_column: str | None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    df = pd.read_csv(input_csv)
    if valid_column and valid_column in df.columns:
        df = df[df[valid_column].astype(bool)].copy()
    df = df.dropna(subset=[id_column, sequence_column])
    if limit is not None:
        df = df.head(limit)
    return [
        (str(row[id_column]), str(row[sequence_column]))
        for _, row in df.iterrows()
    ]


def build_embeddings(
    *,
    dataset: str,
    input_csv: Path,
    output_dir: Path,
    id_column: str,
    sequence_column: str,
    valid_column: str | None,
    model_name: str,
    device: str,
    hf_home: Path,
    limit: int | None = None,
    weights_path: str | None = None,
    resume: bool = False,
) -> None:
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str((hf_home / "hub").resolve())
    input_before = sha256_file(input_csv) if resume else None
    records = load_records(
        input_csv,
        id_column=id_column,
        sequence_column=sequence_column,
        valid_column=valid_column,
        limit=limit,
    )
    if resume:
        if sha256_file(input_csv) != input_before:
            raise ValueError('Input changed while records were being loaded')
        if not weights_path:
            raise ValueError('--resume requires an explicit --esm-checkpoint')
        return build_resumable(dataset=dataset, input_csv=input_csv, output_dir=output_dir,
                               records=records, model_name=model_name, device=device,
                               weights_path=weights_path, selection={'id_column':id_column,
                               'sequence_column':sequence_column,'valid_column':valid_column,'limit':limit})
    encoder = ESM3Encoder(model_name=model_name, device=device, weights_path=weights_path)
    embeddings = []
    for idx, (record_id, sequence) in enumerate(records, start=1):
        print(f"[{dataset}] ESM3 encoding {idx}/{len(records)}: {record_id}", flush=True)
        embeddings.append(encoder.encode_sequence(record_id, sequence))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_embeddings_npz(
        embeddings,
        output_npz=output_dir / f"{dataset}.npz",
        metadata_csv=output_dir / f"{dataset}_metadata.csv",
        metadata_json=output_dir / f"{dataset}_metadata.json",
        model_name=model_name,
    )
    print(f"ESM3 embeddings written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIG) + ["custom"],
        required=True,
        help="Processed dataset to embed.",
    )
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default="outputs/esm3_embeddings")
    parser.add_argument("--model-name", default="esm3-sm-open-v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-home", default="outputs/hf_cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="Only verify that the required ESM3 package can be imported.",
    )
    parser.add_argument("--resume", action="store_true", help="Persist per-sequence cache and resume only identical input/checkpoint jobs")
    parser.add_argument("--esm-checkpoint", help="Pinned snapshot data/weights/esm3_sm_open_v1.pth")
    parser.add_argument("--id-column")
    parser.add_argument("--sequence-column")
    parser.add_argument("--valid-column")
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    if args.dataset == "custom" and not all((args.input_csv, args.id_column, args.sequence_column, args.output_prefix)):
        parser.error("custom requires --input-csv --id-column --sequence-column --output-prefix")
    if args.output_prefix and (Path(args.output_prefix).name != args.output_prefix or args.output_prefix in (".","..")):
        parser.error("output prefix must be a filename stem")

    if args.check_dependencies:
        os.environ["HF_HOME"] = str(Path(args.hf_home).resolve())
        os.environ["HUGGINGFACE_HUB_CACHE"] = str((Path(args.hf_home) / "hub").resolve())
        try:
            ESM3Encoder(model_name=args.model_name, device=args.device, weights_path=args.esm_checkpoint)
        except ESM3SetupError as exc:
            raise SystemExit(str(exc)) from exc
        print("ESM3 dependency and checkpoint are available.")
        return

    config = dict(DATASET_CONFIG.get(args.dataset, {"prefix":args.output_prefix,"input":args.input_csv,"id_column":args.id_column,"sequence_column":args.sequence_column,"valid_column":None}))
    for option in ("id_column","sequence_column","valid_column"):
        if getattr(args,option) is not None: config[option]=getattr(args,option)
    if args.output_prefix: config["prefix"]=args.output_prefix
    build_embeddings(
        dataset=config["prefix"],
        input_csv=Path(args.input_csv or config["input"]),
        output_dir=Path(args.output_dir),
        id_column=config["id_column"],
        sequence_column=config["sequence_column"],
        valid_column=config["valid_column"],
        model_name=args.model_name,
        device=args.device,
        hf_home=Path(args.hf_home),
        limit=args.limit,
        weights_path=args.esm_checkpoint,
        resume=args.resume,
    )




def _atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _entry_key(record_id, sequence, binding):
    return hashlib.sha256(json.dumps([record_id, sequence, binding], sort_keys=True).encode()).hexdigest()


def _read_entry(path, key, record_id, sequence):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['cache_key'].item()) != key or str(saved['record_id'].item()) != record_id or str(saved['sequence'].item()) != sequence:
            raise ValueError(f'Cache identity mismatch: {path}')
        residue, global_vector = saved['residue'], saved['global_embedding']
        if residue.ndim != 2 or residue.shape[0] != len(sequence) or global_vector.shape != (residue.shape[1],):
            raise ValueError(f'Invalid cached embedding shapes: {path}')
        if not np.isfinite(residue).all() or not np.isfinite(global_vector).all():
            raise ValueError(f'Nonfinite cached embedding: {path}')
        return residue, global_vector


def _write_entry(path, key, item):
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, cache_key=key, record_id=item.record_id, sequence=item.sequence,
                                residue=np.asarray(item.residue_embedding, dtype=np.float32),
                                global_embedding=np.asarray(item.global_embedding, dtype=np.float32))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def _assemble_streaming(records, entries, output_dir, dataset, binding):
    """Write residue arrays one at a time and global vectors through a disk memmap."""
    with tempfile.TemporaryDirectory(prefix='.' + dataset + '-assemble-', dir=output_dir) as temp:
        temporary = Path(temp)
        first_residue, first_global = _read_entry(entries[0][0], entries[0][1], *records[0]) if records else (None, np.empty(0))
        dimension = len(first_global)
        del first_residue, first_global
        global_matrix = np.lib.format.open_memmap(temporary/'global.npy', mode='w+', dtype=np.float32, shape=(len(records), dimension)) if records else np.empty((0, 0), dtype=np.float32)
        rows = []
        with zipfile.ZipFile(temporary/f'{dataset}.npz', 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for index, ((record_id, sequence), (path, key)) in enumerate(zip(records, entries)):
                residue, vector = _read_entry(path, key, record_id, sequence)
                if len(vector) != dimension:
                    raise ValueError('Embedding dimensions differ across cache entries')
                global_matrix[index] = vector
                embedding_key = f'emb_{index:08d}'
                with archive.open(embedding_key + '.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, residue, allow_pickle=False)
                rows.append({'record_id':record_id,'sequence':sequence,'length':len(sequence),
                             'embedding_key':embedding_key,'global_index':index})
                del residue, vector
            if records: global_matrix.flush()
            with archive.open('global_embeddings.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array(member, global_matrix, allow_pickle=False)
        del global_matrix
        pd.DataFrame(rows, columns=['record_id','sequence','length','embedding_key','global_index']).to_csv(temporary/f'{dataset}_metadata.csv', index=False)
        metadata = {'schema_version':1,'encoder':'ESM3','model_name':binding['model_name'],
                    'record_count':len(records),'embedding_dimension':dimension,'status':'complete',
                    'binding':binding,'npz':str(output_dir/f'{dataset}.npz'),
                    'metadata_csv':str(output_dir/f'{dataset}_metadata.csv')}
        metadata['output_sha256'] = {p.name:sha256_file(p) for p in [temporary/f'{dataset}.npz',temporary/f'{dataset}_metadata.csv']}
        for name in metadata['output_sha256']:
            os.replace(temporary/name, output_dir/name)
        _atomic_json(output_dir/f'{dataset}_metadata.json', metadata)


def _runtime_fingerprint(weights_path):
    """Bind cache semantics, encoder implementation, dependencies and snapshot assets."""
    versions = {}
    for distribution in ('esm', 'torch', 'numpy', 'transformers', 'tokenizers'):
        try:
            versions[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            versions[distribution] = None
    weight = Path(weights_path).absolute()
    companions = {}
    # The explicit ESM3 loader uses the snapshot's companion tokenizer/resources.
    # Exclude weight files: the active encoder checkpoint is hashed separately;
    # lazy structure/function decoders are not used for sequence-only encoding.
    if weight.parent.name == 'weights' and weight.parent.parent.name == 'data':
        snapshot = weight.parent.parent.parent
        for item in sorted(snapshot.rglob('*')):
            if item.is_file() and 'weights' not in item.relative_to(snapshot).parts:
                companions[str(item.relative_to(snapshot))] = sha256_file(item)
    return {
        'cache_format_version': 2,
        'pooling': 'aligned_biological_residue_mean_float32_v1',
        'encoder_source_sha256': sha256_file(Path(__file__).with_name('esm3_encoder.py')),
        'python': platform.python_version(),
        'packages': versions,
        'snapshot_companion_sha256': companions,
    }


def build_resumable(*, dataset, input_csv, output_dir, records, model_name, device, weights_path, selection):
    """Bind incremental files to exact input bytes and pinned model bytes.

    A successfully atomically written sequence survives any later inference error.
    Changed input, model or selection settings must use a new output prefix/cache.
    """
    import fcntl
    if Path(dataset).name != dataset or dataset in ('.','..'):
        raise ValueError('Dataset must be a filename stem')
    if len({record_id for record_id, _ in records}) != len(records):
        raise ValueError('Resumable embeddings require unique record IDs')
    weights = Path(weights_path)
    if not weights.is_file(): raise FileNotFoundError(weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / ('_cache_' + dataset)
    cache.mkdir(exist_ok=True)
    binding = {'input_sha256':sha256_file(input_csv),'checkpoint_sha256':sha256_file(weights),
               'model_name':model_name,'device':device,'dataset':dataset,'selection':selection,
               'runtime_fingerprint':_runtime_fingerprint(weights),
               'records_sha256':hashlib.sha256(json.dumps(records).encode()).hexdigest()}
    with (cache/'.lock').open('a+') as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError(f'Embedding cache already in use: {cache}') from exc
        manifest_path = cache/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('binding') != binding:
                raise ValueError('Embedding cache input/model/settings changed; use a new output prefix or directory')
        else:
            if any((output_dir / name).exists() for name in (f'{dataset}.npz', f'{dataset}_metadata.csv', f'{dataset}_metadata.json')):
                raise ValueError('Existing final embeddings have no resumable binding manifest; use a new output prefix')
            if any(p.name != '.lock' for p in cache.iterdir()):
                raise ValueError('Existing cache has no binding manifest')
            manifest = {'schema_version':1,'binding':binding,'record_count':len(records),'completed_records':0}
        manifest.update(status='running', input_csv=str(input_csv.resolve()), checkpoint=str(weights.absolute()))
        manifest.pop('error', None)
        _atomic_json(manifest_path, manifest)
        prior_completed = manifest.get('completed_records', 0)
        encoder, entries = None, []
        try:
            for index, (record_id, sequence) in enumerate(records):
                key = _entry_key(record_id, sequence, binding)
                path = cache/(key+'.npz')
                receipt = path.with_suffix('.json')
                if index < prior_completed and (not path.exists() or not receipt.exists()):
                    raise ValueError('Previously completed cache entry or hash receipt is missing')
                if path.exists() and receipt.exists():
                    saved_receipt = json.loads(receipt.read_text())
                    if saved_receipt.get('cache_key') != key or saved_receipt.get('sha256') != sha256_file(path):
                        raise ValueError(f'Cached embedding changed: {path}')
                    _read_entry(path, key, record_id, sequence)
                    print(f'[{dataset}] cached {index+1}/{len(records)}: {record_id}', flush=True)
                else:
                    if encoder is None:
                        encoder = ESM3Encoder(model_name=model_name, device=device, weights_path=str(weights))
                    print(f'[{dataset}] encoding {index+1}/{len(records)}: {record_id}', flush=True)
                    item = encoder.encode_sequence(record_id, sequence)
                    if item.record_id != record_id or item.sequence != sequence:
                        raise ValueError('Encoder returned a different record identity')
                    _write_entry(path, key, item)
                    _read_entry(path, key, record_id, sequence)
                    _atomic_json(receipt, {'cache_key':key,'sha256':sha256_file(path)})
                    del item
                entries.append((path, key))
                manifest['completed_records'] = max(prior_completed, index + 1)
                _atomic_json(manifest_path, manifest)
            encoder = None
            if sha256_file(input_csv) != binding['input_sha256'] or sha256_file(weights) != binding['checkpoint_sha256']:
                raise ValueError('Input or model changed during encoding')
            if _runtime_fingerprint(weights) != binding['runtime_fingerprint']:
                raise ValueError('Encoder runtime or companion model resources changed during encoding')
            print(f'[{dataset}] assembling {len(records)} cached embeddings', flush=True)
            _assemble_streaming(records, entries, output_dir, dataset, binding)
            manifest['status'] = 'complete'
            _atomic_json(manifest_path, manifest)
            print(f'ESM3 embeddings written to {output_dir}', flush=True)
        except BaseException as exc:
            manifest.update(status='failed', error=str(exc))
            _atomic_json(manifest_path, manifest)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == '__main__':
    main()
