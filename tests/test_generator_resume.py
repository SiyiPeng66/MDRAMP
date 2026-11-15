"""Tiny synthetic sequence fixtures test epoch-boundary exact CPU resume."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import torch


def command(tmp_path, output, epochs, resume=False, extra=()):
    argv=[sys.executable,'-m','src.generation.cli','pretrain','--data',str(tmp_path/'data.csv'),'--output',str(tmp_path/output),'--epochs',str(epochs),'--seed','11','--hidden','16','--heads','2','--layers','1','--feedforward','32','--dropout','.2','--batch-size','2','--patience','10','--training-log',str(tmp_path/(output+'.jsonl'))]
    if resume:argv+=['--resume-training']
    return subprocess.run([*argv,*extra],capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},timeout=60)


def test_exact_epoch_resume_and_bound_inputs(tmp_path):
    pd.DataFrame({'sequence':['KALWQKALWQ','RALWQKALWQ','GALWQKALWQ','AALWQKALWQ'],'split':['train','train','train','validation'],'cluster_id':['a','b','c','d']}).to_csv(tmp_path/'data.csv',index=False)
    continuous=command(tmp_path,'continuous.pt',3);assert continuous.returncode==0,continuous.stderr
    first=command(tmp_path,'resumed.pt',1);assert first.returncode==0,first.stderr
    assert 'epoch=1/1' in first.stdout and 'batch=2/2' in first.stdout
    resumed=command(tmp_path,'resumed.pt',3,True);assert resumed.returncode==0,resumed.stderr
    a=torch.load(tmp_path/'continuous.training.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed.training.pt',weights_only=True)
    assert a['next_epoch']==b['next_epoch']==3
    for key,value in a['model'].items():assert torch.equal(value,b['model'][key]),key
    assert a['best']==b['best'] and a['stale']==b['stale']
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    records=[json.loads(line) for line in (tmp_path/'resumed.pt.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in records]==[1,2,3]
    mismatch=command(tmp_path,'resumed.pt',4,True,extra=['--learning-rate','.002'])
    assert mismatch.returncode!=0 and 'mismatch' in mismatch.stderr
    with (tmp_path/'data.csv').open('a') as f:f.write('\n')
    changed=command(tmp_path,'resumed.pt',4,True)
    assert changed.returncode!=0 and 'mismatch' in changed.stderr


def test_output_protection_paths_runtime_and_lock(tmp_path):
    import fcntl
    pd.DataFrame({'sequence':['KALWQKALWQ','RALWQKALWQ'],'split':['train','validation'],'cluster_id':['a','b']}).to_csv(tmp_path/'data.csv',index=False)
    existing=tmp_path/'existing.pt';existing.write_bytes(b'other run')
    refused=command(tmp_path,'existing.pt',1)
    assert refused.returncode!=0 and 'already exist' in refused.stderr
    assert existing.read_bytes()==b'other run'
    log=tmp_path/'logonly.pt.jsonl';log.write_text('existing log\n')
    refused=command(tmp_path,'logonly.pt',1)
    assert refused.returncode!=0 and log.read_text()=='existing log\n'
    run=command(tmp_path,'owned.pt',1);assert run.returncode==0,run.stderr
    renamed=command(tmp_path,'new.pt',2,True,extra=['--training-state',str(tmp_path/'owned.training.pt')])
    assert renamed.returncode!=0 and 'mismatch' in renamed.stderr
    assert not (tmp_path/'new.pt').exists()
    statepath=tmp_path/'owned.training.pt'
    state=torch.load(statepath,weights_only=True)
    original=statepath.read_bytes()
    state['binding']['runtime']['torch']='different-version'
    torch.save(state,statepath)
    invalid=command(tmp_path,'owned.pt',2,True)
    assert invalid.returncode!=0 and 'mismatch' in invalid.stderr
    statepath.write_bytes(original)
    state=torch.load(statepath,weights_only=True)
    state['binding']['implementation']['policy.py']='0'*64
    torch.save(state,statepath)
    invalid=command(tmp_path,'owned.pt',2,True)
    assert invalid.returncode!=0 and 'mismatch' in invalid.stderr
    statepath.write_bytes(original)
    with (tmp_path/'owned.pt.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=command(tmp_path,'owned.pt',2,True,extra=['--training-state',str(tmp_path/'alternative.training.pt')])
        assert locked.returncode!=0 and 'locked' in locked.stderr
    best=tmp_path/'owned.pt';best.write_bytes(b'changed externally')
    invalid=command(tmp_path,'owned.pt',2,True)
    assert invalid.returncode!=0 and 'hash mismatch' in invalid.stderr
    assert best.read_bytes()==b'changed externally'
