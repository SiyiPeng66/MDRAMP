"""Derive explicitly hypothetical MIC design targets without editing measurements."""
import argparse
import json
import math
from pathlib import Path
import shutil

import pandas as pd

from mdramp.calibration_train import fresh_output, sha


def derive(dataset_dir, output_dir, factor):
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError('Target divisor must be finite and greater than one')
    source = Path(dataset_dir).resolve()
    manifest_path = source / 'manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('target_semantics', 'measured_mic') != 'measured_mic':
        raise ValueError('Cannot rescale an already derived target dataset')
    required = ('observations.csv', 'sequences.csv')
    hashes = manifest['output_sha256']
    for name in required:
        if name not in hashes or sha(source / name) != hashes[name]:
            raise ValueError('Source dataset hash mismatch: ' + name)
    obs = pd.read_csv(source / 'observations.csv')
    if 'target_semantics' in obs or any(c.startswith('measured_') for c in obs):
        raise ValueError('Cannot rescale an already derived target dataset')
    if not obs.status.eq('usable').all():
        raise ValueError('Resolve missing/quarantined observations before deriving targets')
    shift = math.log2(factor)
    for column in ('lower_log2', 'upper_log2', 'exact_log2'):
        values = pd.to_numeric(obs[column], errors='raise')
        obs['measured_' + column] = values
        obs[column] = values - shift
    if obs[['lower_log2', 'upper_log2']].isna().any().any():
        raise ValueError('Missing bounds')
    if (obs.lower_log2 > obs.upper_log2).any():
        raise ValueError('Invalid bounds')
    obs['target_semantics'] = 'design_target_not_measured'
    obs['target_divisor'] = factor
    implementation_hash = sha(__file__)
    with fresh_output(output_dir) as out:
        obs.to_csv(out / 'observations.csv', index=False)
        shutil.copyfile(source / 'sequences.csv', out / 'sequences.csv')
        shutil.copyfile(manifest_path, out / 'source_manifest.json')
        if sha(manifest_path) != manifest_hash or any(sha(source / n) != hashes[n] for n in required):
            raise ValueError('Source dataset changed during derivation')
        if sha(__file__) != implementation_hash:
            raise ValueError('Implementation changed during derivation')
        document = {
            'schema_version': 1, 'purpose': 'hypothetical_design_target_training',
            'target_semantics': 'design_target_not_measured', 'target_divisor': factor,
            'transformation': 'target_log2 = measured_log2 - log2(target_divisor)',
            'log2_shift': shift, 'source_dataset': str(source),
            'source_manifest_sha256': manifest_hash,
            'source_input_sha256': {n: hashes[n] for n in required},
            'implementation_sha256': implementation_hash,
            'observations': len(obs), 'unique_sequences': int(obs.peptide_id.nunique()),
            'historical_prospective_reproduction': False,
            'limitations': [
                'These are user-requested design targets, not new experimental measurements.',
                'Censoring, inclusivity, sequence clusters and folds are preserved.',
                'Smaller target predictions do not establish improved biological activity.',
            ],
            'output_sha256': {n: sha(out / n) for n in ('observations.csv', 'sequences.csv', 'source_manifest.json')},
        }
        (out / 'manifest.json').write_text(json.dumps(document, indent=2, allow_nan=False))
    print(json.dumps({'observations': len(obs), 'target_divisor': factor, 'target_semantics': document['target_semantics']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--factor', type=float, required=True)
    args = parser.parse_args()
    derive(args.dataset_dir, args.output_dir, args.factor)


if __name__ == '__main__':
    main()
