"""Validate manuscript generation prerequisites without loading ESM3 for inference."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scorer-config', required=True)
    parser.add_argument('--output', help='Optional new JSON readiness report')
    args = parser.parse_args()
    from src.score.frozen_scorer import FrozenSequenceScorer
    report = {'scorer_config': str(Path(args.scorer_config).resolve()),
              'scope': 'artifact and manuscript-policy validation, not an end-to-end inference test'}
    try:
        scorer = FrozenSequenceScorer(args.scorer_config)
        report.update(ready=True, validation=scorer.validate_artifacts())
    except (ValueError, KeyError, TypeError, OSError) as error:
        report.update(ready=False, error=str(error))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as handle:
            handle.write(text + '\n')
    print(text)
    if not report['ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
