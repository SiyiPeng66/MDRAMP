"""Aggregate a complete measured six-strain panel into its minimum MIC interval."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

STRAINS = ('Ab_WT', 'Sa_WT', 'CRAB', 'MRSA', 'CRKP', 'CRPA')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def boolean(value):
    if str(value).lower() in ('true', '1'): return True
    if str(value).lower() in ('false', '0'): return False
    raise ValueError(f'Invalid interval endpoint flag: {value!r}')


def minimum_interval(rows):
    """Exact image of the Cartesian product of intervals under minimum."""
    bounds = []
    for row in rows:
        lo, hi = float(row['lower_log2']), float(row['upper_log2'])
        li, ui = boolean(row['lower_inclusive']), boolean(row['upper_inclusive'])
        if math.isnan(lo) or math.isnan(hi) or lo == math.inf or hi == -math.inf or lo > hi or (lo == hi and not (li and ui)):
            raise ValueError('Invalid or empty source MIC interval')
        bounds.append((lo, hi, li and math.isfinite(lo), ui and math.isfinite(hi)))
    if not bounds: raise ValueError('Empty panel')
    lower = min(row[0] for row in bounds)
    upper = min(row[1] for row in bounds)
    lower_inclusive = any(row[2] for row in bounds if row[0] == lower)
    upper_inclusive = all(row[3] for row in bounds if row[1] == upper)
    return lower, upper, lower_inclusive, upper_inclusive


def aggregate(sequences, observations):
    if any(str(column).startswith('measured_') for column in observations):
        raise ValueError('Only measured source panels are accepted; aggregate before target transformation')
    if 'target_semantics' in observations and not observations.target_semantics.eq('measured_mic').all():
        raise ValueError('Only measured source panels are accepted')
    if 'target_divisor' in observations and not pd.to_numeric(observations.target_divisor, errors='raise').eq(1).all():
        raise ValueError('Measured source panels require target_divisor=1')
    if sequences.empty or sequences.peptide_id.duplicated().any() or sequences.sequence.duplicated().any():
        raise ValueError('Sequences must be nonempty and unique')
    if set(sequences.peptide_id) != set(observations.peptide_id):
        raise ValueError('Sequence and observation IDs differ')
    result = []
    for sequence in sequences.to_dict('records'):
        panel = observations.loc[observations.peptide_id == sequence['peptide_id']]
        if len(panel) != 6 or set(panel.strain_id) != set(STRAINS) or not panel.status.eq('usable').all():
            raise ValueError(f"Complete usable six-strain panel required: {sequence['peptide_id']}")
        for field in ('cluster_id', 'fold'):
            if field in panel and not panel[field].eq(sequence[field]).all():
                raise ValueError(f'Inconsistent panel {field}')
        panel = panel.set_index('strain_id').loc[list(STRAINS)].reset_index()
        lo, hi, li, ui = minimum_interval(panel.to_dict('records'))
        kind = ('exact' if lo == hi and li and ui else 'left_censored' if lo == -math.inf else 'right_censored' if hi == math.inf else 'interval')
        # JSON-oriented pandas conversion represents missing numbers as null.
        evidence = json.loads(panel.to_json(orient='records'))
        for raw, saved in zip(panel.to_dict('records'), evidence):
            for field in ('lower_log2', 'upper_log2'):
                if math.isinf(float(raw[field])): saved[field] = 'inf' if raw[field] > 0 else '-inf'
        result.append(dict(peptide_id=sequence['peptide_id'], strain_id='best_panel', lower_log2=lo,
                           upper_log2=hi, lower_inclusive=li, upper_inclusive=ui, status='usable',
                           observation_type=kind, exact_log2=lo if kind == 'exact' else None,
                           cluster_id=sequence['cluster_id'], fold=sequence['fold'], source_strain_count=6,
                           provenance=json.dumps(evidence, ensure_ascii=False, allow_nan=False), exclusion_reason=''))
    return pd.DataFrame(result)


def prepare(dataset_dir, output_dir):
    source, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output.exists(): raise ValueError('Output already exists; choose a new version')
    code = {str(Path(__file__).resolve()): sha256(__file__)}
    source_manifest = source / 'manifest.json'
    manifest_hash = sha256(source_manifest)
    original = json.loads(source_manifest.read_text())
    if original.get('target_semantics', 'measured_mic') != 'measured_mic' or original.get('target_divisor', 1) != 1:
        raise ValueError('Only measured source panels are accepted; aggregate before target transformation')
    recorded = original.get('output_sha256', {})
    if not {'sequences.csv', 'observations.csv'} <= set(recorded):
        raise ValueError('Source manifest must bind sequences and observations')
    inputs = {str(source_manifest): manifest_hash}
    for name, expected in recorded.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source): raise ValueError('Source manifest path escapes dataset')
        if sha256(path) != expected: raise ValueError(f'Source hash mismatch: {name}')
        inputs[str(path)] = expected
    sequences = pd.read_csv(source / 'sequences.csv')
    result = aggregate(sequences, pd.read_csv(source / 'observations.csv'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists(): raise ValueError('Output already exists')
        staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
        try:
            shutil.copyfile(source / 'sequences.csv', staging / 'sequences.csv')
            shutil.copyfile(source_manifest, staging / 'source_manifest.json')
            result.to_csv(staging / 'observations.csv', index=False)
            for path, expected in {**inputs, **code}.items():
                if sha256(path) != expected: raise ValueError(f'Input or implementation changed: {path}')
            manifest = dict(schema_version=1, purpose='retrospective_minimum_MIC_across_measured_panel',
                            historical_prospective_reproduction=False, target_semantics='measured_mic', target_divisor=1, strains=['best_panel'],
                            source_strains=list(STRAINS), mic_units=original.get('mic_units'),
                            aggregation='minimum over complete six-strain MIC intervals; lower=min(lower), upper=min(upper); tied lower endpoint any inclusive, tied upper endpoint all inclusive',
                            interpretation='Minimum only within this measured panel; screened intervals remain censored unless the minimum interval collapses to an exact value.',
                            fold_count=original.get('fold_count'), seed=original.get('seed'),
                            grouping=original.get('grouping'), unique_sequences=len(sequences),
                            observation_type_counts=result.observation_type.value_counts().to_dict(),
                            inputs_sha256=inputs, implementation_sha256=code,
                            output_sha256={name:sha256(staging / name) for name in ('sequences.csv','observations.csv','source_manifest.json')})
            (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            if output.exists(): raise ValueError('Output appeared during preparation')
            os.rename(staging, output)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    manifest = prepare(args.dataset_dir, args.output_dir)
    print(json.dumps({'unique_sequences':manifest['unique_sequences'], 'observation_type_counts':manifest['observation_type_counts']}))


if __name__ == '__main__': main()
