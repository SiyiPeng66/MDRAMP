"""Versioned, resumable MDRAMP subprocess workflows with artifact provenance."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import time
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ALLOWED_MODULES = frozenset({
    'src.data.build_processed', 'src.data.parse_public', 'src.data.parse_private_round',
    'src.data.parse_targets', 'src.features.build_esm3_embeddings',
    'src.features.physicochemical', 'src.external.pore_forming_wrapper',
    'src.external.tpeppro_wrapper', 'src.external.convert_tpeppro_output',
    'src.train.pretrain_public', 'src.train.train_target_affinity',
    'src.train.train_mic_head', 'src.score.predict_target_affinity',
    'src.score.score_candidates', 'src.score.predict', 'src.score.contribution_analysis',
    'src.score.evaluate_public',
    'src.data.prepare_tpeppro_structures', 'src.features.prott5_embeddings',
    'src.external.tpeppro_public', 'src.data.screen_public_pairs',
    'src.selection.freeze_round', 'src.selection.r5_safety',
    'src.generation.cli', 'src.generation.pairs', 'src.loop.model_state',
})
REPO = Path(__file__).resolve().parents[2]

class WorkflowError(RuntimeError):
    """An invalid workflow, changed artifact or failed stage."""

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def artifact_hash(path):
    """Hash file bytes or a directory's sorted relative paths and file hashes."""
    path = Path(path)
    if path.is_symlink():
        raise WorkflowError(f'Symlink artifacts are unsupported: {path}')
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return {'kind': 'file', 'sha256': digest.hexdigest()}
    if path.is_dir():
        entries = {}
        for item in sorted(path.rglob('*')):
            if item.is_symlink():
                raise WorkflowError(f'Symlink artifacts are unsupported: {item}')
            if item.is_file():
                entries[str(item.relative_to(path))] = artifact_hash(item)['sha256']
            elif item.is_dir():
                entries[str(item.relative_to(path)) + '/'] = None
        return {'kind': 'directory', 'sha256': canonical_hash(entries)}
    raise WorkflowError(f'Missing artifact: {path}')

def snapshot(paths, existing_only=False):
    return {p: artifact_hash(p) for p in paths if not existing_only or Path(p).exists()}

def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def exact_keys(mapping, allowed, required, label):
    if not isinstance(mapping, dict):
        raise WorkflowError(f'{label} must be an object')
    missing, extra = set(required) - set(mapping), set(mapping) - set(allowed)
    if missing or extra:
        raise WorkflowError(f'{label}: missing={sorted(missing)}, unknown={sorted(extra)}')

def load_config(path):
    path = Path(path).resolve()
    if path.suffix.lower() in {'.yaml', '.yml'}:
        import yaml
        config = yaml.safe_load(path.read_text())
    else:
        config = json.loads(path.read_text())
    exact_keys(config, {'schema_version', 'workspace', 'steps'}, {'schema_version', 'steps'}, 'workflow')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise WorkflowError('Only schema_version 1 is supported')
    workspace_value = config.get('workspace', '.')
    if not isinstance(workspace_value, str):
        raise WorkflowError('workspace must be a path string')
    workspace = (path.parent / workspace_value).resolve()
    if not workspace.is_dir():
        raise WorkflowError(f'Workspace does not exist: {workspace}')
    if not isinstance(config['steps'], list) or not config['steps']:
        raise WorkflowError('steps must be a nonempty list')
    normalized, names, produced = [], set(), set()
    for step in config['steps']:
        exact_keys(step, {'name', 'module', 'args', 'inputs', 'outputs'}, {'name', 'module', 'args', 'inputs', 'outputs'}, 'stage')
        name = step['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name) or name in names:
            raise WorkflowError('Stage names must be unique safe identifiers')
        names.add(name)
        if step['module'] not in ALLOWED_MODULES:
            raise WorkflowError(f'Module is not an allowed stage: {step["module"]}')
        normalized_step = {'name': name, 'module': step['module']}
        for key in ('inputs', 'outputs'):
            values = step[key]
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise WorkflowError(f'{name}.{key} must be a list of nonempty paths')
            paths = [str((workspace / v).resolve()) for v in values]
            if len(paths) != len(set(paths)):
                raise WorkflowError(f'Duplicate {key} in stage {name}')
            normalized_step[key] = paths
        if not normalized_step['outputs']:
            raise WorkflowError(f'Stage {name} must declare outputs')
        for output in normalized_step['outputs']:
            if output in produced or any(output == inp or Path(output) in Path(inp).parents or Path(inp) in Path(output).parents for inp in normalized_step['inputs']):
                raise WorkflowError(f'Output cannot overwrite an input or prior stage output: {output}')
            for prior in produced:
                if Path(output) in Path(prior).parents or Path(prior) in Path(output).parents:
                    raise WorkflowError('Stage outputs cannot overlap across stages')
            produced.add(output)
        argv = step['args']
        if not isinstance(argv, list):
            raise WorkflowError(f'{name}.args must be an argv list')
        args = []
        for value in argv:
            if isinstance(value, dict):
                exact_keys(value, {'path'}, {'path'}, 'path argument')
                if not isinstance(value['path'], str):
                    raise WorkflowError('path argument must be a string')
                resolved = str((workspace / value['path']).resolve())
                if resolved not in normalized_step['inputs'] + normalized_step['outputs']:
                    raise WorkflowError('Typed path arguments must be declared inputs or outputs')
                args.append(resolved)
            elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                args.append(str(value))
            else:
                raise WorkflowError('Arguments must be scalar tokens or {path: ...}')
        normalized_step['args'] = args
        normalized.append(normalized_step)
    # Forward references are invalid; a workflow is explicitly sequential.
    all_outputs = {p for s in normalized for p in s['outputs']}
    available = set()
    for step in normalized:
        for p in step['inputs']:
            if p in all_outputs and p not in available:
                raise WorkflowError(f'Input refers to a later stage output: {p}')
        available.update(step['outputs'])
    return {'schema_version': 1, 'workspace': str(workspace), 'steps': normalized}

@contextlib.contextmanager
def exclusive_lock(run_dir):
    import fcntl
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError(f'Workflow is already running: {run_dir}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def plan(config):
    return [{'name': s['name'], 'cwd': config['workspace'],
             'argv': [sys.executable, '-m', s['module'], *s['args']],
             'inputs': s['inputs'], 'outputs': s['outputs']} for s in config['steps']]

def _retry_argv(stage, planned_argv, previous, *, resume, workspace):
    """Resume only audited producer output/state, without changing config."""
    argv = list(planned_argv)
    if not resume or not previous or previous.get('status') not in {'failed', 'running'}:
        return argv
    args = stage['args']
    if stage['module'] in {'src.data.prepare_tpeppro_structures',
                           'src.features.prott5_embeddings', 'src.external.tpeppro_public'}:
        values=[]
        for index,value in enumerate(args):
            if value=='--output-dir':
                if index+1>=len(args):raise WorkflowError('--output-dir is missing its argument')
                values.append(args[index+1])
            elif value.startswith('--output-dir='):
                values.append(value.split('=',1)[1])
        if len(values)!=1 or not values[0]:
            raise WorkflowError('Resumable producer must declare exactly one output-dir')
        output=str((Path(workspace)/values[0]).resolve())
        if not Path(output).exists():return argv
        if (not Path(output).is_dir() or output not in stage['outputs']
                or output not in previous.get('partial_outputs',{})):
            raise WorkflowError('Cannot automatically resume an undeclared or unaudited producer output')
        if artifact_hash(output)!=previous['partial_outputs'][output]:
            raise WorkflowError('Producer output changed after audit')
        if '--resume' not in args:argv.append('--resume')
        return argv
    if stage['module'] != 'src.generation.cli' or not args or args[0] != 'pretrain':
        return argv
    if '--resume-training' in args:
        return argv
    values = []
    for index, value in enumerate(args):
        if value == '--training-state':
            if index + 1 >= len(args):
                raise WorkflowError('--training-state is missing its argument')
            values.append(args[index + 1])
        elif value.startswith('--training-state='):
            values.append(value.split('=', 1)[1])
    if not values:
        return argv  # Never infer the generator's default state path.
    if len(values) != 1:
        raise WorkflowError('Generator stage must declare exactly one training-state option')
    state = str((Path(workspace) / values[0]).resolve())
    if not Path(state).is_file():
        return argv
    if state not in stage['outputs'] or state not in previous.get('partial_outputs', {}):
        raise WorkflowError('Cannot automatically resume an undeclared or unaudited generator training state')
    if artifact_hash(state) != previous['partial_outputs'][state]:
        raise WorkflowError('Generator training state changed after audit')
    argv.append('--resume-training')
    return argv


def wait_for_stage_inputs(stage, manifest, manifest_path, timeout):
    """Wait only before an unstarted stage has captured any input identity."""
    deadline=time.monotonic()+timeout
    since=now()
    try:
        while True:
            missing=[p for p in stage['inputs'] if not Path(p).exists()]
            if not missing:
                manifest['status']='running'
                manifest.pop('waiting_for',None)
                manifest.pop('error',None)
                atomic_json(manifest_path,manifest)
                return
            manifest['status']='waiting'
            manifest['waiting_for']={'stage':stage['name'],'missing_paths':missing,'since':since}
            atomic_json(manifest_path,manifest)
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise WorkflowError(f'Timed out waiting for inputs of {stage["name"]}: {missing}')
            time.sleep(min(1.0,remaining))
    except BaseException as exc:
        manifest['error']=str(exc) or type(exc).__name__
        atomic_json(manifest_path,manifest)
        raise


def run(config_path, run_dir, resume=False, dry_run=False, wait_for_inputs=0):
    if not math.isfinite(wait_for_inputs) or wait_for_inputs<0:
        raise WorkflowError('wait-for-inputs must be a finite nonnegative number')
    config = load_config(config_path)
    planned = plan(config)
    if dry_run:
        return {'schema_version': 1, 'status': 'planned', 'steps': planned}
    run_dir = Path(run_dir).resolve()
    for stage in config['steps']:
        for p in stage['outputs']:
            output = Path(p)
            if output == run_dir or output in run_dir.parents or run_dir in output.parents:
                raise WorkflowError('Stage outputs must not overlap the workflow run directory')
    manifest_path = run_dir / 'workflow.json'
    config_hash = canonical_hash(config)
    runtime = {'python': sys.version, 'executable': sys.executable,
               'packages': {name: importlib.metadata.version(name) for name in ('numpy','pandas','torch','pyyaml')},
               'source_sha256': canonical_hash({str(p.relative_to(REPO)): artifact_hash(p)['sha256'] for p in sorted((REPO / 'src').rglob('*.py'))})}
    with exclusive_lock(run_dir):
        if manifest_path.exists():
            if not resume:
                raise WorkflowError('Run exists; use --resume explicitly or a new run directory')
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('runtime') != runtime:
                raise WorkflowError('Cannot resume: Python runtime or source code changed')
            if manifest.get('config_sha256') != config_hash:
                raise WorkflowError('Cannot resume: normalized workflow configuration changed')
            for i, record in enumerate(manifest['steps']):
                if record['status'] == 'completed':
                    if snapshot(config['steps'][i]['inputs']) != record['inputs'] or snapshot(config['steps'][i]['outputs']) != record['outputs']:
                        raise WorkflowError(f'Cannot resume: artifacts changed for {record["name"]}')
                elif record['status'] in {'failed', 'running'}:
                    if record.get('partial_outputs_error'):
                        raise WorkflowError('Cannot resume: failed output artifacts could not be audited')
                    if snapshot(config['steps'][i]['inputs']) != record.get('inputs'):
                        raise WorkflowError(f'Cannot resume: failed-stage inputs changed for {record["name"]}')
                    partial = snapshot(config['steps'][i]['outputs'], existing_only=True)
                    if partial != record.get('partial_outputs', {}):
                        raise WorkflowError('Cannot resume: partial outputs changed or interrupted write is unaudited')
        else:
            if resume:
                raise WorkflowError('Cannot resume: no prior manifest')
            if any(p.name != '.lock' for p in run_dir.iterdir()):
                raise WorkflowError('Run directory is not empty')
            for stage in config['steps']:
                if any(Path(p).exists() for p in stage['outputs']):
                    raise WorkflowError(f'Refusing to overwrite existing output of {stage["name"]}')
            manifest = {'schema_version': 1, 'config_sha256': config_hash, 'config': config,
                        'status': 'running', 'created_at': now(), 'runtime': runtime, 'steps': []}
            atomic_json(manifest_path, manifest)
        manifest['status'] = 'running'
        manifest.pop('error', None)
        atomic_json(manifest_path, manifest)
        for i, stage in enumerate(config['steps']):
            if i < len(manifest['steps']) and manifest['steps'][i]['status'] == 'completed':
                continue
            previous = manifest['steps'][i] if i < len(manifest['steps']) else None
            if previous is None and wait_for_inputs>0:
                wait_for_stage_inputs(stage,manifest,manifest_path,wait_for_inputs)
            actual_argv = _retry_argv(stage, planned[i]['argv'], previous, resume=resume, workspace=config['workspace'])
            attempt = (previous.get('attempt', 0) + 1) if previous else 1
            record = {'name': stage['name'], 'status': 'running', 'attempt': attempt,
                      'started_at': now(), 'argv': actual_argv, 'planned_argv': planned[i]['argv'], 'inputs': {}}
            if i < len(manifest['steps']):
                manifest['steps'][i] = record
            else:
                manifest['steps'].append(record)
            try:
                record['inputs'] = snapshot(stage['inputs'])
                atomic_json(manifest_path, manifest)
                log_path = run_dir / f'{i:03d}-{stage["name"]}-attempt{attempt}.log'
                temporary = log_path.with_suffix('.log.running')
                record['log'] = str(log_path)
                env = dict(os.environ)
                env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
                try:
                    with temporary.open('x') as log:
                        process = subprocess.run(record['argv'], cwd=config['workspace'], env=env,
                                                 stdout=log, stderr=subprocess.STDOUT, check=False)
                        log.flush()
                        os.fsync(log.fileno())
                    record['returncode'] = process.returncode
                finally:
                    if temporary.exists():
                        os.replace(temporary, log_path)
                if process.returncode:
                    raise WorkflowError(f'Stage {stage["name"]} failed with exit code {process.returncode}; see {log_path}')
                if snapshot(stage['inputs']) != record['inputs']:
                    raise WorkflowError(f'Stage {stage["name"]} mutated its declared inputs')
                record['outputs'] = snapshot(stage['outputs'])
                record['status'] = 'completed'
                record['completed_at'] = now()
                atomic_json(manifest_path, manifest)
            except BaseException as exc:
                record['status'] = 'failed'
                record['error'] = str(exc)
                record['completed_at'] = now()
                try:
                    record['partial_outputs'] = snapshot(stage['outputs'], existing_only=True)
                except Exception as hash_error:
                    record['partial_outputs'] = {}
                    record['partial_outputs_error'] = str(hash_error)
                manifest['status'] = 'failed'
                manifest['error'] = str(exc)
                atomic_json(manifest_path, manifest)
                raise
        manifest['status'] = 'completed'
        manifest['completed_at'] = now()
        atomic_json(manifest_path, manifest)
        return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-for-inputs',type=float,default=0,metavar='SECONDS',help='Wait for missing inputs before starting each new stage')
    parser.add_argument('--dry-run', '--plan', dest='dry_run', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and not args.run_dir:
        parser.error('--run-dir is required for execution')
    result = run(args.config, args.run_dir, args.resume, args.dry_run, args.wait_for_inputs)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
