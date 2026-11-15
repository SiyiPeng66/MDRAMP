import pytest
from mdramp.prospective_workflow import retry_argv
from src.workflow.run import artifact_hash,WorkflowError


def test_rl_resume_requires_unchanged_declared_state(tmp_path):
    path=tmp_path/'state.pt';path.write_bytes(b'audited fixture')
    args=['reinforce','--training-state',str(path)]
    stage={'module':'src.generation.cli','args':args,'outputs':[str(path)]}
    planned=['python','-m','src.generation.cli',*args]
    previous={'status':'failed','partial_outputs':{str(path):artifact_hash(str(path))}}
    result=retry_argv(stage,planned,previous,resume=True,workspace=str(tmp_path))
    assert result==planned+['--resume-training']
    assert 'reinforce' in result and 'pretrain' not in result
    assert retry_argv(stage,planned,None,resume=False,workspace=str(tmp_path))==planned
    path.write_bytes(b'changed')
    with pytest.raises(WorkflowError,match='changed'):
        retry_argv(stage,planned,previous,resume=True,workspace=str(tmp_path))
