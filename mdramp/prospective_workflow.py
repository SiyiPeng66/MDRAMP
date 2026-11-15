"""Audited workflow router for new prospective model development."""
from src.workflow import run


_original_retry=run._retry_argv

def retry_argv(stage,planned_argv,previous,*,resume,workspace):
    # Apply the existing exact declared/audited generator-state checks to RL.
    # The executed argv retains reinforce; only the helper's mode guard is shared.
    if stage['module']=='src.generation.cli' and stage['args'] and stage['args'][0]=='reinforce':
        stage={**stage,'args':['pretrain',*stage['args'][1:]]}
    elif stage['module']=='mdramp.panel_pair_cache':
        # Reuse the same audited output-directory checks; executed module stays unchanged.
        stage={**stage,'module':'src.features.prott5_embeddings'}
    return _original_retry(stage,planned_argv,previous,resume=resume,workspace=workspace)


def main():
    run.ALLOWED_MODULES = run.ALLOWED_MODULES | frozenset({
        'mdramp.prospective_receipt','mdramp.prospective_panelseal','mdramp.prospective_features',
        'mdramp.prospective_calibration','mdramp.prospective_candidates',
        'mdramp.panel_pair_cache','mdramp.panel_stability'})
    run._retry_argv=retry_argv
    run.main()

if __name__=='__main__':main()
