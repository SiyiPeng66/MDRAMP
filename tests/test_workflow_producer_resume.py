from types import SimpleNamespace
import json
import pytest
import src.workflow.run as w

MODULES=['src.data.prepare_tpeppro_structures','src.features.prott5_embeddings','src.external.tpeppro_public']

@pytest.mark.parametrize('module',MODULES)
def test_audited_producer_retry(tmp_path,monkeypatch,module):
    source=tmp_path/'input.csv';source.write_text('data')
    config={'schema_version':1,'workspace':'.','steps':[{'name':'producer','module':module,
        'args':['--output-dir',{'path':'output'}],'inputs':['input.csv'],'outputs':['output']}]}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));commands=[]
    def process(argv,**kwargs):
        commands.append(list(argv));out=tmp_path/'output';out.mkdir(exist_ok=True)
        (out/'binding.json').write_text('identity')
        if len(commands)==1:return SimpleNamespace(returncode=1)
        assert argv[-1]=='--resume'
        (out/'done').write_text('complete');return SimpleNamespace(returncode=0)
    monkeypatch.setattr(w.subprocess,'run',process)
    with pytest.raises(w.WorkflowError,match='exit code'):w.run(path,tmp_path/'run')
    result=w.run(path,tmp_path/'run',resume=True)
    assert result['steps'][0]['attempt']==2 and '--resume' not in commands[0]
    assert '--resume' not in result['steps'][0]['planned_argv']


def test_no_fresh_adoption_or_unaudited_directory(tmp_path):
    output=tmp_path/'output';output.mkdir();(output/'binding').write_text('a')
    stage={'module':MODULES[0],'args':['--output-dir=output'],'outputs':[str(output)]}
    planned=['python',*stage['args']]
    previous={'status':'failed','partial_outputs':{str(output):w.artifact_hash(output)}}
    assert w._retry_argv(stage,planned,None,resume=True,workspace=tmp_path)==planned
    assert w._retry_argv(stage,planned,previous,resume=False,workspace=tmp_path)==planned
    with pytest.raises(w.WorkflowError,match='unaudited'):
        w._retry_argv(stage,planned,{'status':'failed'},resume=True,workspace=tmp_path)
    with pytest.raises(w.WorkflowError,match='unaudited'):
        w._retry_argv({**stage,'outputs':[]},planned,previous,resume=True,workspace=tmp_path)
    (output/'binding').write_text('changed')
    with pytest.raises(w.WorkflowError,match='changed'):
        w._retry_argv(stage,planned,previous,resume=True,workspace=tmp_path)
