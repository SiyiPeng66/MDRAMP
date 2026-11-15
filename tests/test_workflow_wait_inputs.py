import importlib
import json
import pytest
from test_workflow_production import config
w=importlib.import_module('src.workflow.run')


def fake_clock(monkeypatch, callback=None):
    clock=[0.]
    monkeypatch.setattr(w.time,'monotonic',lambda:clock[0])
    def sleep(seconds):
        assert 0<seconds<=1
        clock[0]+=seconds
        if callback:callback()
    monkeypatch.setattr(w.time,'sleep',sleep)


def test_wait_persists_before_stage_and_runs_after_arrival(tmp_path,monkeypatch):
    cfg=config(tmp_path);source=tmp_path/'input.csv';source.unlink();run_dir=tmp_path/'run'
    def arrive():
        state=json.loads((run_dir/'workflow.json').read_text())
        assert state['status']=='waiting' and state['steps']==[]
        assert state['waiting_for']['missing_paths']==[str(source)]
        source.write_text('sequence\nKALWQKALWQ\n')
    fake_clock(monkeypatch,arrive)
    result=w.run(cfg,run_dir,wait_for_inputs=10)
    assert result['status']=='completed' and 'waiting_for' not in result
    assert result['steps'][0]['inputs']


def test_timeout_and_resume_unstarted(tmp_path,monkeypatch):
    cfg=config(tmp_path);source=tmp_path/'input.csv';source.unlink();run_dir=tmp_path/'run'
    fake_clock(monkeypatch)
    with pytest.raises(w.WorkflowError,match='Timed out'):w.run(cfg,run_dir,wait_for_inputs=2)
    state=json.loads((run_dir/'workflow.json').read_text())
    assert state['status']=='waiting' and state['steps']==[]
    source.write_text('sequence\nKALWQKALWQ\n')
    result=w.run(cfg,run_dir,resume=True,wait_for_inputs=2)
    assert result['status']=='completed' and result['steps'][0]['attempt']==1


def test_interrupt_preserves_unstarted_stage(tmp_path,monkeypatch):
    cfg=config(tmp_path);(tmp_path/'input.csv').unlink();run_dir=tmp_path/'run'
    def interrupt():raise KeyboardInterrupt()
    fake_clock(monkeypatch,interrupt)
    with pytest.raises(KeyboardInterrupt):w.run(cfg,run_dir,wait_for_inputs=2)
    state=json.loads((run_dir/'workflow.json').read_text())
    assert state['status']=='waiting' and state['steps']==[]


def test_wait_zero_retains_failed_stage_and_cannot_bypass_changed_inputs(tmp_path):
    cfg=config(tmp_path);source=tmp_path/'input.csv';source.unlink();run_dir=tmp_path/'run'
    with pytest.raises(w.WorkflowError,match='Missing artifact'):w.run(cfg,run_dir)
    state=json.loads((run_dir/'workflow.json').read_text())
    assert state['steps'][0]['status']=='failed'
    source.write_text('sequence\nKALWQKALWQ\n')
    with pytest.raises(w.WorkflowError,match='failed-stage inputs changed'):
        w.run(cfg,run_dir,resume=True,wait_for_inputs=100)


def test_resume_rechecks_completed_stage_before_waiting(tmp_path,monkeypatch):
    cfg=config(tmp_path);payload=json.loads(cfg.read_text())
    payload['steps'].append({'name':'later','module':'src.features.physicochemical',
       'args':['--input',{'path':'later.csv'},'--output',{'path':'later_features.csv'}],
       'inputs':['later.csv'],'outputs':['later_features.csv']})
    cfg.write_text(json.dumps(payload));fake_clock(monkeypatch)
    with pytest.raises(w.WorkflowError,match='Timed out'):
        w.run(cfg,tmp_path/'run',wait_for_inputs=1)
    state=json.loads((tmp_path/'run/workflow.json').read_text())
    assert len(state['steps'])==1 and state['steps'][0]['status']=='completed'
    (tmp_path/'input.csv').write_text('changed')
    with pytest.raises(w.WorkflowError,match='artifacts changed'):
        w.run(cfg,tmp_path/'run',resume=True,wait_for_inputs=1)
