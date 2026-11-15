import json
import os
from pathlib import Path
import pytest
from src.workflow.run import WorkflowError, run, exclusive_lock, load_config


def config(tmp_path, bad=False):
    (tmp_path/'input.csv').write_text('sequence\nKALWQKALWQ\n')
    payload={'schema_version':1,'workspace':'.','steps':[
        {'name':'features','module':'src.features.physicochemical',
         'args':['--input',{'path':'input.csv'},'--output',{'path':'features.csv'}]+(['--invalid'] if bad else []),
         'inputs':['input.csv'],'outputs':['features.csv']}]}
    path=tmp_path/'workflow.json';path.write_text(json.dumps(payload));return path


def test_real_cli_run_resume_and_changed_artifact(tmp_path):
    path=config(tmp_path);out=tmp_path/'run'
    result=run(path,out)
    assert result['status']=='completed'
    assert result['steps'][0]['returncode']==0
    assert (tmp_path/'features.csv').exists()
    assert run(path,out,resume=True)['steps'][0]['attempt']==1
    with pytest.raises(WorkflowError,match='Run exists'):run(path,out)
    (tmp_path/'features.csv').write_text('changed')
    with pytest.raises(WorkflowError,match='artifacts changed'):run(path,out,resume=True)


def test_missing_outputs_marks_failure(tmp_path):
    path=config(tmp_path);payload=json.loads(path.read_text())
    payload['steps'][0]['outputs']=['features.csv','absent.csv'];path.write_text(json.dumps(payload))
    with pytest.raises(WorkflowError,match='Missing artifact'):run(path,tmp_path/'run')
    record=json.loads((tmp_path/'run/workflow.json').read_text())
    assert record['status']=='failed'
    assert record['steps'][0]['returncode']==0


def test_child_failure_atomic_logs_and_manifest(tmp_path):
    path=config(tmp_path,bad=True)
    with pytest.raises(WorkflowError,match='exit code'):run(path,tmp_path/'run')
    state=json.loads((tmp_path/'run/workflow.json').read_text())
    assert state['steps'][0]['status']=='failed'
    assert Path(state['steps'][0]['log']).exists()
    with pytest.raises(WorkflowError):run(path,tmp_path/'run',resume=True)
    state=json.loads((tmp_path/'run/workflow.json').read_text())
    assert state['steps'][0]['attempt']==2


def test_plan_no_writes_and_strict_allowlist(tmp_path):
    path=config(tmp_path)
    assert run(path,None,dry_run=True)['status']=='planned'
    assert not (tmp_path/'features.csv').exists()
    data=json.loads(path.read_text());data['steps'][0]['module']='src.train.finetune_round'
    path.write_text(json.dumps(data))
    with pytest.raises(WorkflowError,match='allowed'):load_config(path)


def test_config_changes_and_existing_outputs_rejected(tmp_path):
    path=config(tmp_path);out=tmp_path/'run';run(path,out)
    data=json.loads(path.read_text());data['steps'][0]['args']+=['--sequence-column','sequence'];path.write_text(json.dumps(data))
    with pytest.raises(WorkflowError,match='configuration changed'):run(path,out,resume=True)
    with pytest.raises(WorkflowError,match='overwrite'):run(path,tmp_path/'other')


def test_concurrent_lock_and_yaml(tmp_path):
    import yaml
    path=config(tmp_path);yaml_path=tmp_path/'workflow.yaml';yaml_path.write_text(yaml.safe_dump(json.loads(path.read_text())))
    assert load_config(path)==load_config(yaml_path)
    with exclusive_lock(tmp_path/'run'):
        with pytest.raises(WorkflowError,match='already running'):
            run(path,tmp_path/'run')


def test_input_change_prevents_resume(tmp_path):
    path=config(tmp_path);out=tmp_path/'run';run(path,out)
    (tmp_path/'input.csv').write_text('sequence\nAAAAAKKKKK\n')
    with pytest.raises(WorkflowError,match='artifacts changed'):run(path,out,resume=True)

def test_containment_and_undeclared_path_rejected(tmp_path):
    path=config(tmp_path);data=json.loads(path.read_text())
    data['steps'][0]['outputs']=['.'];data['steps'][0]['args']=['--help']
    path.write_text(json.dumps(data))
    with pytest.raises(WorkflowError,match='overwrite'):load_config(path)
    path=config(tmp_path);data=json.loads(path.read_text())
    data['steps'][0]['args'] += ['--other',{'path':'undeclared'}]
    path.write_text(json.dumps(data))
    with pytest.raises(WorkflowError,match='declared'):load_config(path)


def test_input_mutation_during_stage_rejected(tmp_path, monkeypatch):
    import src.workflow.run as workflow
    from types import SimpleNamespace
    path=config(tmp_path)
    def mutate(*args,**kwargs):
        (tmp_path/'input.csv').write_text('sequence\nCHANGED\n')
        (tmp_path/'features.csv').write_text('output')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(workflow.subprocess,'run',mutate)
    with pytest.raises(WorkflowError,match='mutated'):run(path,tmp_path/'run')
    manifest=json.loads((tmp_path/'run/workflow.json').read_text())
    assert manifest['status']=='failed'
