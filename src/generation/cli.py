"""Train, fine-tune, and sample the V6 generator (python -m src.generation.cli)."""
import argparse
import importlib
import random
import os
import json
import tempfile
import copy
import time
import fcntl
import platform
from contextlib import contextmanager, ExitStack
from pathlib import Path
import pandas as pd
import torch
from src.generation.policy import Policy, reinforce_loss
from src.generation.runtime import SUPPORT,validate_training_frame
from src.utils.hash import sha256_file
import sys
from src.selection.filters import passes_guardrails
from src.optimization.pep_b_genetic import edit_distance

def save(model, path, conditional, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(dict(config=model.config, state_dict=model.state_dict(), conditional=conditional, metadata=metadata), path)

def atomic_torch_save(payload, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            torch.save(payload,handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def write_training_log(path, records):
    if not path: return
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            for record in records: handle.write(json.dumps(record)+'\n')
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


@contextmanager
def pretrain_output_lock(args):
    """Lock all canonical destinations, including shared best paths across states."""
    output=Path(args.output).resolve()
    state=Path(args.training_state).resolve() if args.training_state else output.with_suffix('.training.pt')
    paths=[output,state]+([Path(args.training_log).resolve()] if args.training_log else [])
    if len(set(paths))!=len(paths): raise ValueError('Training destinations must be distinct')
    with ExitStack() as stack:
        for path in sorted(paths):
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=stack.enter_context(path.with_name(path.name+'.lock').open('a'))
            try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc: raise ValueError(f'Training destination is locked: {path}') from exc
        if not args.resume_training:
            existing=[str(path) for path in paths if path.exists()]
            if existing: raise ValueError(f'Training destinations already exist: {existing}; resume or choose new paths')
        yield


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Policy(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model, ckpt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['pretrain','reinforce','sample'])
    p.add_argument('--output', required=True); p.add_argument('--checkpoint'); p.add_argument('--reference')
    p.add_argument('--data'); p.add_argument('--conditional', action='store_true'); p.add_argument('--parent')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device', default='cpu'); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=256); p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=4); p.add_argument('--feedforward', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--steps', type=int); p.add_argument('--beta-kl', type=float)
    p.add_argument('--scorer', default='src.score.frozen_scorer:FrozenSequenceScorer', help='module:function factory accepting scorer config path')
    p.add_argument('--scorer-config'); p.add_argument('--reference-sequences', help='CSV sequence column; excluded homologues outside local parent family')
    p.add_argument('--protected-sequences', help='Independent test/challenge sequences; never exempted by parent-family membership')
    p.add_argument('--eligible-output', help='Optional unique eligible candidate-pool CSV; requires reference coverage')
    p.add_argument('--parent-family', help='Explicit permitted parent-family CSV sequence column')
    p.add_argument('--count', type=int, default=200); p.add_argument('--top-p', type=float, default=.95)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--training-log', help='Epoch JSONL metrics, reconstructed consistently on resume')
    p.add_argument('--training-state', help='Last completed epoch checkpoint; defaults to output with .training.pt suffix')
    p.add_argument('--resume-training', action='store_true', help='Resume supervised epochs or RL steps from the last atomic state')
    a=p.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    if a.threads<1:p.error('positive threads required')
    torch.set_num_threads(a.threads)
    if a.batch_size < 1 or a.epochs < 1 or a.patience < 1: p.error('positive batch/epoch/patience required')
    if a.action == 'pretrain':
        with pretrain_output_lock(a):
            if not a.data: p.error('--data required')
            if a.learning_rate <= 0 or a.weight_decay < 0: p.error('Invalid optimizer hyperparameters')
            data_hash=sha256_file(a.data)
            frame=pd.read_csv(a.data)
            validate_training_frame(frame,a.conditional)
            required={'sequence','split','cluster_id'} | ({'parent'} if a.conditional else set())
            if not required.issubset(frame): raise ValueError(f'Required columns: {required}')
            if (frame.groupby('cluster_id')['split'].nunique()>1).any(): raise ValueError('Cluster split leakage')
            train=frame[frame.split.eq('train')]; val=frame[frame.split.eq('validation')]
            if train.empty or val.empty: raise ValueError('Train and validation splits required')
            model=Policy(a.hidden,a.heads,a.layers,a.feedforward,a.dropout).to(a.device)
            optimizer=torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
            state_path=Path(a.training_state) if a.training_state else Path(a.output).with_suffix('.training.pt')
            if state_path.resolve()==Path(a.output).resolve(): raise ValueError('Training state and best output must be distinct')
            if Path(a.data).resolve() in {state_path.resolve(),Path(a.output).resolve()}: raise ValueError('Checkpoints must not overwrite input data')
            if a.training_log and Path(a.training_log).resolve() in {state_path.resolve(),Path(a.output).resolve(),Path(a.data).resolve()}: raise ValueError('Training log must not overwrite input/checkpoints')
            binding={key:getattr(a,key) for key in ('conditional','seed','batch_size','learning_rate','weight_decay','hidden','heads','layers','feedforward','dropout','patience','device','threads')}
            binding['data_sha256']=data_hash
            binding['destinations']={'output':str(Path(a.output).resolve()),'state':str(state_path.resolve()),'log':str(Path(a.training_log).resolve()) if a.training_log else None}
            binding['implementation']={name:sha256_file(Path(__file__).with_name(name)) for name in ('cli.py','policy.py','runtime.py')}
            binding['runtime']={'python':platform.python_version(),'torch':str(torch.__version__)}
            best=float('inf'); stale=0; first_epoch=0; history=[]; best_payload=None
            if a.resume_training:
                state=torch.load(state_path,map_location='cpu',weights_only=True)
                if state.get('schema_version')!=1 or state.get('binding')!=binding: raise ValueError('Training state input/configuration mismatch')
                if not Path(a.output).is_file() or sha256_file(a.output)!=state.get('best_sha256'):
                    raise ValueError('Best checkpoint hash mismatch; refusing to overwrite changed output')
                model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                first_epoch=state['next_epoch']; best=state['best']; stale=state['stale']; history=state['history']; best_payload=state['best_payload']
                if a.epochs < first_epoch: raise ValueError('Requested epochs precede completed training state')
                torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
                if a.device.startswith('cuda'): torch.cuda.set_rng_state_all(state['cuda_rng'])
                if a.device.startswith('mps'): torch.mps.set_rng_state(state['mps_rng'])
                write_training_log(a.training_log,history)
                print(f'pretrain resume: completed={first_epoch} best_validation_nll={best:.6f} stale={stale}',flush=True)
            elif state_path.exists():
                raise ValueError('Training state already exists; use --resume-training or a new output')
            print(f'pretrain start: train={len(train)} validation={len(val)} epochs={a.epochs} device={a.device}',flush=True)
            for epoch in range(first_epoch,a.epochs):
                if stale>=a.patience: break
                started=time.monotonic(); model.train(); train_total=0.; train_tokens=0
                order=torch.randperm(len(train)).tolist(); batches=(len(train)+a.batch_size-1)//a.batch_size
                for batch_index,start in enumerate(range(0,len(train),a.batch_size),1):
                    batch=train.iloc[order[start:start+a.batch_size]]
                    loss=model.supervised_loss(batch.sequence.tolist(),batch.parent.tolist() if a.conditional else None)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    n=sum(len(seq)+1 for seq in batch.sequence);train_total+=float(loss.detach())*n;train_tokens+=n
                    if batch_index%25==0 or batch_index==batches:
                        print(f'epoch={epoch+1}/{a.epochs} batch={batch_index}/{batches} train_loss={train_total/train_tokens:.6f}',flush=True)
                model.eval(); total=0.; tokens=0
                with torch.no_grad():
                    for start in range(0,len(val),a.batch_size):
                        batch=val.iloc[start:start+a.batch_size]; n=sum(len(seq)+1 for seq in batch.sequence)
                        total+=model.supervised_loss(batch.sequence.tolist(),batch.parent.tolist() if a.conditional else None).item()*n;tokens+=n
                score=total/tokens
                if not torch.isfinite(torch.tensor(score)): raise ValueError('Nonfinite validation loss')
                if sha256_file(a.data)!=data_hash: raise ValueError('Training input changed; epoch checkpoint not published')
                if score<best:
                    best=score; stale=0
                    best_payload=copy.deepcopy(dict(config=model.config,state_dict=model.state_dict(),conditional=a.conditional,metadata=dict(stage='supervised_pretraining',legal_support=SUPPORT,validation_nll=score,epoch=epoch,seed=a.seed,arguments=vars(a),input_hashes={'data':data_hash})))
                    atomic_torch_save(best_payload,a.output)
                else: stale+=1
                record={'epoch':epoch+1,'train_loss':train_total/train_tokens,'validation_nll':score,'best_validation_nll':best,'stale':stale,'elapsed_seconds':time.monotonic()-started}
                history.append(record)
                state=dict(schema_version=1,binding=binding,best_sha256=sha256_file(a.output),model=model.state_dict(),optimizer=optimizer.state_dict(),next_epoch=epoch+1,best=best,stale=stale,history=history,best_payload=best_payload,torch_rng=torch.get_rng_state(),python_rng=random.getstate())
                if a.device.startswith('cuda'): state['cuda_rng']=torch.cuda.get_rng_state_all()
                if a.device.startswith('mps'): state['mps_rng']=torch.mps.get_rng_state()
                atomic_torch_save(state,state_path);write_training_log(a.training_log,history)
                print(f'epoch={epoch+1}/{a.epochs} validation_nll={score:.6f} best={best:.6f} stale={stale}/{a.patience} seconds={record["elapsed_seconds"]:.1f}',flush=True)
            if sha256_file(a.data)!=data_hash: raise ValueError('Training input changed during pretraining')
            return
    if not a.checkpoint: p.error('--checkpoint required')
    model, ckpt=load(a.checkpoint,a.device); model.eval()
    if ckpt['conditional'] != (a.parent is not None): p.error('Conditional checkpoint requires --parent; unconditional forbids it')
    if a.action=='sample':
        from src.generation.runtime import sample
        sample(a,model,ckpt)
        return
    if not a.reference or not a.scorer or not a.scorer_config or a.steps is None or a.beta_kl is None: p.error('RL requires --reference --scorer --scorer-config --steps --beta-kl')
    if a.steps<1 or a.beta_kl<0: p.error('Invalid RL steps/beta')
    for flag in ('--learning-rate','--batch-size','--weight-decay'):
        if not any(v == flag or v.startswith(flag+'=') for v in sys.argv[1:]): p.error('RL requires explicit '+flag)
    if not a.reference_sequences: p.error('RL requires --reference-sequences')
    if a.parent is not None and not a.parent_family: p.error('Conditional RL requires --parent-family')
    if a.parent is None and a.parent_family: p.error('--parent-family only applies to conditional RL')
    from src.generation.runtime import reinforce
    reinforce(a,model,ckpt,load)

if __name__=='__main__': main()
