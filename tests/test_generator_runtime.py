import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import pandas as pd
import pytest
import torch
from src.generation.policy import Policy
from src.generation.cli import load,save
from src.generation.runtime import SUPPORT,assess,eligibility_context,reference_identity,reinforce,sample,validate_training_frame


def args_for(tmp_path,name='result.pt'):
    return SimpleNamespace(output=str(tmp_path/name),checkpoint=str(tmp_path/'reference.pt'),reference=str(tmp_path/'reference.pt'),scorer_config=str(tmp_path/'scorer.json'),reference_sequences=str(tmp_path/'references.csv'),parent_family=None,protected_sequences=None,parent=None,device='cpu',seed=11,learning_rate=.001,weight_decay=0.,steps=3,beta_kl=.1,batch_size=8,training_state=None,training_log=None,resume_training=False,scorer='fixture_generator_scorer:Factory',eligible_output=None,count=4,top_p=.95,temperature=1.)


def fixture(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(11)
    model=Policy(hidden=8,heads=2,layers=1,feedforward=16,dropout=.1)
    save(model,tmp_path/'reference.pt',False,validation_nll=2.,epoch=1,arguments={'action':'pretrain'},input_hashes={'data':'fixture'})
    (tmp_path/'scorer.json').write_text('{}')
    pd.DataFrame({'sequence':['DDDDDDDDDD'],'split':['test']}).to_csv(tmp_path/'references.csv',index=False)
    module=types.ModuleType('fixture_generator_scorer')
    class Factory:
        hashes={}
        fail=False
        def __init__(self,path):pass
        def __call__(self,seqs):
            if self.fail:raise RuntimeError('fixture scoring failure')
            return [.8 if seq.startswith('K') else .2 for seq in seqs]
    module.Factory=Factory;monkeypatch.setitem(sys.modules,'fixture_generator_scorer',module)
    def sampled(self,*args):return ['KALWQKALWQ','RALWQKALWQ'][torch.randint(2,()).item()]
    monkeypatch.setattr(Policy,'sample',sampled)
    return Factory


def execute(args):
    torch.manual_seed(11)
    model,ckpt=load(args.checkpoint,args.device)
    reinforce(args,model,ckpt,load)


def test_reference_and_protected_family(tmp_path):
    metadata={'validation_nll':2.,'epoch':1,'arguments':{'action':'pretrain'},'input_hashes':{'data':'abc'}}
    assert reference_identity({'metadata':metadata})['legacy_metadata_verified']
    with pytest.raises(ValueError,match='supervised'):reference_identity({'metadata':{**metadata,'arguments':{'action':'reinforce'}}})
    with pytest.raises(ValueError,match='support'):reference_identity({'metadata':{**metadata,'legal_support':{}}})
    parent='KALWQKALWQ';args=args_for(tmp_path);args.parent=parent;args.parent_family=str(tmp_path/'family.csv')
    pd.DataFrame({'sequence':[parent],'split':['test']}).to_csv(args.reference_sequences,index=False)
    pd.DataFrame({'sequence':[parent]}).to_csv(args.parent_family,index=False)
    refs=eligibility_context(args,True)
    assert parent in refs
    assert assess('RALWQKALWQ',parent,refs)==(False,'homology_or_previous_testing')


def test_rl_exact_resume_atomic_logs_and_failure(tmp_path,monkeypatch):
    factory=fixture(tmp_path,monkeypatch)
    continuous=args_for(tmp_path,'continuous.pt');execute(continuous)
    partial=args_for(tmp_path,'partial.pt');partial.steps=1;execute(partial)
    partial.steps=3;partial.resume_training=True;execute(partial)
    a=torch.load(tmp_path/'continuous.rl.training.pt',weights_only=True)
    b=torch.load(tmp_path/'partial.rl.training.pt',weights_only=True)
    assert a['next_step']==b['next_step']==3
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    logs=[json.loads(row) for row in Path(partial.output+'.jsonl').read_text().splitlines()]
    assert len(logs)==3 and all(len(row['trajectories'])==8 for row in logs)
    assert all(t['scoring_status']=='complete' for t in logs[0]['trajectories'])
    changed=copy.copy(partial);changed.learning_rate=.002
    with pytest.raises(ValueError,match='mismatch'):execute(changed)
    # Extending a completed policy may fail after a new step; the old published
    # checkpoint remains hash-bound until the extension is fully published.
    original_call=factory.__call__;calls=[]
    def interrupted(self,sequences):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('extension interrupted')
        return original_call(self,sequences)
    partial.steps=5
    with monkeypatch.context() as context:
        context.setattr(factory,'__call__',interrupted)
        with pytest.raises(RuntimeError,match='extension interrupted'):execute(partial)
    interrupted_state=torch.load(tmp_path/'partial.rl.training.pt',weights_only=True)
    assert interrupted_state['next_step']==4 and interrupted_state['output_sha256']
    execute(partial)
    assert torch.load(tmp_path/'partial.rl.training.pt',weights_only=True)['next_step']==5
    factory.fail=True;failed=args_for(tmp_path,'failed.pt')
    with pytest.raises(RuntimeError,match='scoring failure'):execute(failed)
    failure=json.loads(Path(failed.output+'.failure.json').read_text())
    assert failure['completed_steps']==0
    assert all(t['scoring_status']=='scoring_failure' for t in failure['failed_trajectories'])
    assert not Path(failed.output).exists()
    factory.fail=False;failed.resume_training=True;execute(failed)
    assert Path(failed.output).exists() and not Path(failed.output+'.failure.json').exists()


def test_sample_audits_dedup_eligibility_and_no_overwrite(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);args=args_for(tmp_path,'proposals.csv');args.count=20;args.eligible_output=str(tmp_path/'eligible.csv')
    model,ckpt=load(args.checkpoint,'cpu');model.eval();sample(args,model,ckpt)
    proposals=pd.read_csv(args.output);pool=pd.read_csv(args.eligible_output)
    assert len(proposals)==20 and len(pool)==2 and not pool.sequence.duplicated().any()
    assert proposals.eligible.all() and proposals.duplicate_proposal.sum()==18
    manifest=json.loads(Path(args.output+'.manifest.json').read_text())
    assert manifest['input_sha256']['checkpoint'] and manifest['eligibility_assessed']
    with pytest.raises(FileExistsError):sample(args,model,ckpt)


def test_conditional_cross_split_identity_and_pair_kind():
    frame=pd.DataFrame({'sequence':['KALWQKALWQ','RALWQKALWQ'],'parent':['RALWQKALWQ','GALWQKALWQ'],'split':['train','validation'],'cluster_id':['a','b'],'pair_kind':['synthetic_reconstruction']*2})
    with pytest.raises(ValueError,match='split leakage'):validate_training_frame(frame,True)
    frame.loc[0,'parent']='DALWQKALWQ';validate_training_frame(frame,True)
    frame.loc[0,'pair_kind']='measured_improvement'
    with pytest.raises(ValueError,match='synthetic'):validate_training_frame(frame,True)
