"""Workflow retry policy tests; do not run a real training job."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import src.workflow.run as workflow


def test_failed_generator_state_resumed_without_changing_configuration(tmp_path, monkeypatch):
    (tmp_path/'data.csv').write_text('fixture')
    config={'schema_version':1,'workspace':'.','steps':[{
        'name':'generator','module':'src.generation.cli',
        'args':['pretrain','--data',{'path':'data.csv'},'--output',{'path':'reference.pt'},
                '--training-state',{'path':'last.pt'},'--seed','1'],
        'inputs':['data.csv'],'outputs':['reference.pt','last.pt']}]}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));original=path.read_bytes()
    commands=[]
    def process(argv,**kwargs):
        commands.append(list(argv))
        (tmp_path/'last.pt').write_bytes(b'completed epoch fixture')
        if len(commands)==1:return SimpleNamespace(returncode=1)
        assert '--resume-training' in argv
        (tmp_path/'reference.pt').write_bytes(b'best checkpoint fixture')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess,'run',process)
    with pytest.raises(workflow.WorkflowError,match='exit code'):
        workflow.run(path,tmp_path/'run')
    result=workflow.run(path,tmp_path/'run',resume=True)
    assert result['status']=='completed'
    assert '--resume-training' not in commands[0]
    record=result['steps'][0]
    assert record['argv'][-1]=='--resume-training'
    assert '--resume-training' not in record['planned_argv']
    assert path.read_bytes()==original
    assert record['attempt']==2


def test_resume_policy_is_narrow_and_requires_audited_state(tmp_path):
    state=tmp_path/'state.pt';state.write_bytes(b'epoch')
    stage={'module':'src.generation.cli','args':['pretrain','--training-state',str(state)],'outputs':[str(state)]}
    planned=['python','-m',stage['module'],*stage['args']]
    previous={'status':'failed','partial_outputs':{str(state):workflow.artifact_hash(state)}}
    assert workflow._retry_argv(stage,planned,previous,resume=False,workspace=tmp_path)==planned
    assert workflow._retry_argv(stage,planned,None,resume=True,workspace=tmp_path)==planned
    sample={**stage,'args':['sample','--training-state',str(state)]}
    assert workflow._retry_argv(sample,planned,previous,resume=True,workspace=tmp_path)==planned
    with pytest.raises(workflow.WorkflowError,match='unaudited'):
        workflow._retry_argv(stage,planned,{'status':'failed'},resume=True,workspace=tmp_path)
    state.write_bytes(b'changed')
    with pytest.raises(workflow.WorkflowError,match='changed'):
        workflow._retry_argv(stage,planned,previous,resume=True,workspace=tmp_path)
