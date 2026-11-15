"""Read recorded progress for public-data jobs without loading model weights."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def status(root):
    root = Path(root).resolve()
    result = {'root': str(root), 'note': 'Recorded file state; not a process-liveness probe.'}
    locations = {
        'generator_workflow': 'outputs/runs/public_generator/workflow.json',
        'conditional_generator_workflow': 'outputs/runs/public_conditional_generator/workflow.json',
        'amp_membrane_workflow': 'outputs/runs/public_amp_membrane_v2/workflow.json',
        'interaction_workflow': 'outputs/runs/public_interaction/workflow.json',
        'public_embeddings': 'outputs/public_embeddings_v2/_cache_public/manifest.json',
        'background_embeddings': 'outputs/public_background_embeddings/_cache_background/manifest.json',
        'pair_peptide_embeddings': 'outputs/public_pair_embeddings/_cache_peptides/manifest.json',
        'pair_protein_embeddings': 'outputs/public_pair_embeddings/_cache_proteins/manifest.json',
        'structure_verification': 'outputs/public_tpeppro_structures_v2/manifest.json',
        'prott5_embeddings': 'outputs/public_tpeppro_embeddings/manifest.json',
        'interaction_teacher': 'outputs/public_tpeppro_scores/manifest.json',
        'screened_pairs': 'outputs/public_pairs_screened/manifest.json',
        'pore_teacher': 'outputs/public_teacher/_cache_pore_scores/manifest.json',
    }
    for key, relative in locations.items():
        path = root / relative
        if not path.exists():
            result[key] = {'recorded_status': 'not_started'}; continue
        document = json.loads(path.read_text())
        item = {'recorded_status': document.get('status'),
                'last_update_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
        for field in ('completed_records', 'record_count', 'error'):
            if field in document: item[field] = document[field]
        if 'steps' in document:
            item['steps'] = [{'name': row['name'], 'status': row['status']} for row in document['steps']]
        result[key] = item
    log = root / 'outputs/public_training/generator/training.jsonl'
    if log.exists():
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if records:
            result['generator_training'] = {'last_epoch': records[-1],
                'best_epoch': min(records, key=lambda row: row['validation_nll'])['epoch']}
    conditional_log = root / 'outputs/public_conditional/generator/training.jsonl'
    if conditional_log.exists():
        records = [json.loads(line) for line in conditional_log.read_text().splitlines() if line.strip()]
        if records:
            result['conditional_generator_training'] = {'last_epoch': records[-1],
                'best_epoch': min(records, key=lambda row: row['validation_nll'])['epoch']}
    proposals = root / 'outputs/public_generation/proposals.csv'
    if proposals.exists():
        with proposals.open() as handle:
            result['unranked_generated_sequences'] = sum(1 for _ in csv.DictReader(handle))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    print(json.dumps(status(args.root), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
