"""Unified production CLI. YAML options are overridden by explicit command flags."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from mdramp import __version__

# Public command -> implementation module and fixed positional arguments.
COMMANDS = {
 'init': ('mdramp.project', []),
 'models': ('src.assets.cli', []),
 'prepare': ('src.data.build_processed', []),
 'prepare public': ('src.data.prepare_replay', []),
 'prepare background': ('src.data.prepare_background', []),
 'prepare pairs': ('src.data.prepare_pairs', []),
 'prepare structures': ('src.data.prepare_tpeppro_structures', []),
 'prepare screened-pairs': ('src.data.screen_public_pairs', []),
 'prepare calibration': ('mdramp.calibration_data', []),
 'prepare calibration-targets': ('mdramp.calibration_targets', []),
 'prepare calibration-screen-labels': ('mdramp.calibration_screen_labels', []),
 'prepare calibration-best': ('mdramp.calibration_best', []),
 'prepare prospective-panel': ('mdramp.prospective_panel', []),
 'prepare protein-library': ('mdramp.protein_library_download', []),
 'prepare experiment-protein-library': ('mdramp.protein_library_assemble', []),
 'prepare protein-panel': ('mdramp.protein_panel_build', []),
 'prepare functional-panels': ('mdramp.functional_panel', []),
 'evaluate functional-panels': ('mdramp.functional_panel_diagnostics', []),
 'evaluate budget-panel': ('mdramp.panel_budget_evaluate', []),
 'score round100': ('mdramp.round100_candidates', ['--stage','score']),
 'prepare round100': ('mdramp.round100_candidates', ['--stage','assemble']),
 'prepare budget-panel': ('mdramp.panel_budget', []),
 'prepare compressed-panels': ('mdramp.panel_compress', []),
 'prepare prospective-panel-embeddings': ('mdramp.prospective_panelseal', []),
 'features prospective': ('mdramp.prospective_features', []),
 'select prospective': ('mdramp.prospective_candidates', []),
 'embed': ('src.features.build_esm3_embeddings', []),
 'embed panel': ('mdramp.panel_embed', []),
 'prepare panel-confirmation-probes': ('mdramp.panel_confirmation_probes', []),
 'prepare panel-probes': ('mdramp.panel_probes', ['prepare']),
 'assemble panel-probes': ('mdramp.panel_probes', ['assemble']),
 'score panel-cache': ('mdramp.panel_pair_cache', []),
 'evaluate panel-stability': ('mdramp.panel_stability', []),
 'embed prott5': ('src.features.prott5_embeddings', []),
 'features': ('src.features.physicochemical', []),
 'teacher pore': ('src.external.pore_forming_wrapper', []),
 'teacher interaction': ('src.external.tpeppro_runtime', []),
 'teacher interaction public': ('src.external.tpeppro_public', []),
 'teacher safety': ('src.external.hemopi_official', []),
 'train experts': ('src.train.pretrain_public', []),
 'train interaction': ('src.train.train_target_affinity', []),
 'train mic': ('src.train.train_mic_head', []),
 'train calibration': ('mdramp.calibration_train', ['train']),
 'train prospective': ('mdramp.prospective_calibration', []),
 'prepare preference-features': ('mdramp.preference_features', []),
 'train preference': ('mdramp.preference_head', ['train']),
 'predict preference': ('mdramp.preference_head', ['predict']),
 'predict preference-sequences': ('mdramp.preference_scorer', []),
 'predict prospective-cached': ('mdramp.prospective_predict_cached', []),
 'explore prospective-scenarios': ('mdramp.prospective_scenarios', []),
 'predict calibration': ('mdramp.calibration_train', ['predict']),
 'train generator': ('src.generation.cli', ['pretrain']),
 'train policy': ('src.generation.cli', ['reinforce']),
 'generate': ('src.generation.cli', ['sample']),
 'generate batch': ('mdramp.batched_sample', []),
 'generation check': ('mdramp.generation_check', []),
 'reconstruction-pairs': ('src.generation.pairs', []),
 'predict': ('src.score.predict', []),
 'evaluate public': ('src.score.evaluate_public', []),
 'score experts': ('src.score.score_candidates', ['--experts-only']),
 'score cached': ('src.score.score_candidates', []),
 'score interaction': ('src.score.predict_target_affinity', []),
 'select freeze': ('src.selection.freeze_round', []),
 'select safety': ('src.selection.r5_safety', []),
 'select pareto': ('src.selection.pareto', []),
 'analyze contributions': ('src.score.contribution_analysis', []),
 'state': ('src.loop.model_state', []),
 'status public': ('mdramp.public_status', []),
 'workflow': ('src.workflow.run', []),
 'workflow prospective': ('mdramp.prospective_workflow', []),
}
PATH_OPTIONS = set('esm-checkpoint directory input input-csv output output-dir root hf-home checkpoint reference data scorer-config reference-sequences parent-family model python model-root registry embeddings embedding-metadata scaler amp-prior membrane target-affinity mic-checkpoint peptide-features peptide-embeddings peptide-embedding-metadata target-embeddings target-embedding-metadata pair-output aggregate-output checkpoint-dir scaler-output positive-features positive-embeddings positive-embedding-metadata background-features background-embeddings background-embedding-metadata pore-scores pair-scores mic-ledger expert-means state-manifest parent-checkpoint target-fasta rawdata-dir candidates features scores validation-scores quotas-json references antibacterial-scores hemopi-model hemopi-official-python contact-dir pairs round-data run-dir'.split())
PATH_OPTIONS.update({'training-log', 'training-state', 'public', 'public-features',
                     'public-embeddings', 'public-embedding-metadata', 'pair-features',
                     'pair-embeddings', 'pair-embedding-metadata', 'interaction-checkpoint',
                     'model-dir', 'curation-manifest', 'cache-dir', 'sequences',
                     'structure-manifest', 'embedding-manifest', 'all-pairs', 'weights',
                     'teacher-manifest', 'public-manifest'})
PATH_OPTIONS.update({'observations', 'embedding-metadata', 'corrections', 'dataset-dir'})
PATH_OPTIONS.update({'eligible-output', 'protected-sequences', 'panel-manifest'})
PATH_OPTIONS.update({'expert-manifest'})
PATH_OPTIONS.update({'policy'})
PATH_OPTIONS.update({'design','table'})
PATH_OPTIONS.update({'proteins', 'known-sequences'})
PATH_OPTIONS.update({'embedding-provenance'})
PATH_OPTIONS.update({'library-dir','rules','essential-records','essential-manifest','mmseqs'})
PATH_OPTIONS.update({'library-manifest'})
PATH_OPTIONS.update({'verify-input','verification-output','generated','private-sequences','protocol','probes','sources','protein-embeddings','protein-metadata','protein-provenance','probe-embeddings','probe-metadata','probe-provenance'})
PATH_OPTIONS.update({'compression-dir','probes-dir'})
PATH_OPTIONS.update({'source-rows','oof-predictions','source-dir'})
PATH_OPTIONS.update({'panel-dir','alignment-cache','compression-dir','panel','old-probes','cohort-dir','eligible-output','reference-sequences','protected-sequences'})


def config_arguments(path: str, command: str, overrides: list[str]) -> list[str]:
    file = Path(path).resolve()
    document = yaml.safe_load(file.read_text())
    if not isinstance(document, dict) or set(document) - {'schema_version','command','options'}:
        raise ValueError('Command configuration permits only schema_version, command and options')
    if document.get('schema_version') != 1 or document.get('command') != command:
        raise ValueError('Configuration schema_version or command mismatch')
    options = document.get('options', {})
    if not isinstance(options, dict): raise ValueError('options must be a mapping')
    options = {str(k).replace('_','-'):v for k,v in options.items()}
    for assignment in overrides:
        key,sep,value = assignment.partition('=')
        if not sep or not key: raise ValueError('--set requires option=value')
        options[key.replace('_','-')] = yaml.safe_load(value)
    result=[]
    for key,value in options.items():
        if not key or key.startswith('-') or not all(c.isalnum() or c=='-' for c in key):
            raise ValueError(f'Invalid option name: {key}')
        if value is None or value is False: continue
        result.append('--'+key)
        if value is True: continue
        values = value if isinstance(value,list) else [value]
        if not values: raise ValueError(f'Empty list for {key}')
        for item in values:
            if isinstance(item,(dict,list)) or item is None: raise ValueError(f'Non-scalar option {key}')
            text = str(item)
            if key in PATH_OPTIONS:
                candidate = file.parent / Path(text).expanduser()
                # ESM resolves tokenizer resources relative to the snapshot path;
                # resolving its weight symlink would discard that directory.
                text = str(candidate.absolute() if key == 'esm-checkpoint' else candidate.resolve())
            result.append(text)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--version']:
        print(__version__); return 0
    if not argv or argv == ['--help'] or argv == ['-h']:
        print('MDRAMP '+__version__+'\nUsage: python main.py COMMAND [--config FILE] [--set option=value] [OPTIONS]\n')
        print('\n'.join('  '+key for key in COMMANDS))
        print('\nUse COMMAND --help for task options. Workflow/model commands own their configuration flags.')
        return 0
    if len(argv) in (1,2) and argv[0] not in COMMANDS and (len(argv)==1 or argv[1] in ('--help','-h')):
        children=[key for key in COMMANDS if key.startswith(argv[0]+' ')]
        if children:
            print('\n'.join('  python main.py '+key+' [OPTIONS]' for key in children)); return 0
    matched = next((key for key in sorted(COMMANDS,key=lambda k:len(k.split()),reverse=True)
                    if argv[:len(key.split())]==key.split()),None)
    if matched is None:
        print('Unknown command; use python main.py --help',file=sys.stderr); return 2
    remainder=argv[len(matched.split()):]
    module,prefix=COMMANDS[matched]
    try:
        if matched not in {'workflow','workflow prospective','models'}:
            parser=argparse.ArgumentParser(add_help=False)
            parser.add_argument('--config'); parser.add_argument('--set',action='append',default=[])
            parser.add_argument('--print-command',action='store_true')
            wrapper,remainder=parser.parse_known_args(remainder)
            if wrapper.set and not wrapper.config: raise ValueError('--set requires --config')
            configured=config_arguments(wrapper.config,matched,wrapper.set) if wrapper.config else []
            command=[sys.executable,'-m',module,*prefix,*configured,*remainder]
            if wrapper.print_command:
                print(json.dumps(command,ensure_ascii=False)); return 0
        else:
            command=[sys.executable,'-m',module,*remainder]
        environment=dict(os.environ)
        # Source-checkout entry point must also work from another directory,
        # without an editable install. Preserve the caller's relative paths.
        root = str(Path(__file__).resolve().parents[1])
        environment['PYTHONPATH'] = os.pathsep.join(
            part for part in (root, environment.get('PYTHONPATH', '')) if part
        )
        environment.setdefault('TOKENIZERS_PARALLELISM','false')
        return subprocess.run(command,env=environment,check=False).returncode
    except (ValueError,OSError,yaml.YAMLError) as error:
        print(f'mdramp: {error}',file=sys.stderr); return 2
    except KeyboardInterrupt:
        return 130

if __name__=='__main__': raise SystemExit(main())
