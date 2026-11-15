"""Audited generation eligibility, fixed references, and resumable policy updates."""
import copy
from contextlib import contextmanager
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import pandas as pd
import torch
from src.generation.policy import encode, reinforce_loss
from src.selection.filters import passes_guardrails
from src.optimization.pep_b_genetic import edit_distance
from src.utils.hash import sha256_file

SUPPORT={'vocabulary':'ACDEFGHIKLMNPQRSTVWY','minimum_residues':10,'maximum_residues':50,'eos_in_probability':True,'eos_before_minimum':False,'at_maximum':'EOS_only'}


def validate_training_frame(frame,conditional=False):
    required={'sequence','split','cluster_id'}|({'parent','pair_kind'} if conditional else set())
    if not required.issubset(frame) or frame[list(required)].isna().any().any():raise ValueError('Missing training fields')
    if not frame.split.isin(['train','validation','test']).all():raise ValueError('Unknown/public-challenge split cannot train generator')
    if (frame.groupby('cluster_id').split.nunique()>1).any() or (frame.groupby('sequence').split.nunique()>1).any():raise ValueError('Cluster or sequence split leakage')
    for sequence in frame.sequence:encode(sequence)
    if conditional:
        if not frame.pair_kind.eq('synthetic_reconstruction').all():raise ValueError('Conditional pretraining requires synthetic reconstruction pairs')
        seen={}
        for row in frame.itertuples():
            encode(row.parent)
            if not 1<=edit_distance(row.parent,row.sequence)<=min(int(.3*len(row.parent)),6) or abs(len(row.parent)-len(row.sequence))>3:raise ValueError('Invalid restricted reconstruction pair')
            for sequence in (row.parent,row.sequence):
                if sequence in seen and seen[sequence]!=row.split:raise ValueError('Conditional parent/target split leakage')
                seen[sequence]=row.split


def reference_identity(checkpoint):
    meta=checkpoint.get('metadata',{});args=meta.get('arguments',{})
    if args.get('action')!='pretrain' or not isinstance(meta.get('epoch'),int) or meta['epoch']<0 or not math.isfinite(float(meta.get('validation_nll',float('nan')))) or not meta.get('input_hashes',{}).get('data'):
        raise ValueError('Reference must be a validation-selected supervised pretraining checkpoint')
    if meta.get('stage','supervised_pretraining')!='supervised_pretraining':raise ValueError('RL checkpoint cannot be reference')
    if 'legal_support' in meta and meta['legal_support']!=SUPPORT:raise ValueError('Reference legal support differs')
    return {'identity':'validation_selected_SFT','legacy_metadata_verified':'legal_support' not in meta,'validation_nll':meta['validation_nll'],'epoch':meta['epoch'],'data_sha256':meta['input_hashes']['data']}


def input_hashes(args):
    return {key:sha256_file(getattr(args,key)) for key in ('checkpoint','reference','scorer_config','reference_sequences','parent_family','protected_sequences') if getattr(args,key,None)}


def verify_inputs(args,hashes):
    if any(sha256_file(getattr(args,key))!=value for key,value in hashes.items()):raise ValueError('Generation input artifact changed')


def eligibility_context(args,required=False):
    if not args.reference_sequences:
        if required:raise ValueError('Eligibility requires --reference-sequences')
        return None
    frame=pd.read_csv(args.reference_sequences)
    if 'sequence' not in frame or frame.empty or frame.sequence.isna().any():raise ValueError('Nonempty reference sequence CSV required')
    references=set(frame.sequence.astype(str));protected=set()
    if 'split' in frame:protected.update(frame.loc[frame.split.isin(['test','challenge','independent_challenge']),'sequence'].astype(str))
    if args.protected_sequences:
        locked=pd.read_csv(args.protected_sequences)
        if 'sequence' not in locked or locked.sequence.isna().any():raise ValueError('Invalid protected sequence CSV')
        protected.update(locked.sequence.astype(str))
    if args.parent:
        encode(args.parent)
        if not args.parent_family:raise ValueError('Conditional eligibility requires registered --parent-family')
        if not args.protected_sequences and 'split' not in frame:raise ValueError('Conditional eligibility requires --protected-sequences or typed split references')
        family_frame=pd.read_csv(args.parent_family)
        if 'sequence' not in family_frame or family_frame.sequence.isna().any():raise ValueError('Invalid registered family')
        family=set(family_frame.sequence.astype(str))
        if args.parent not in family:raise ValueError('Registered family must include parent')
        for sequence in family:encode(sequence)
        references-=family
    elif args.parent_family:raise ValueError('Parent family is conditional only')
    references|=protected  # Test/challenge protection always takes precedence over family exception.
    return sorted(references)


def assess(sequence,parent,references):
    if parent:
        distance=edit_distance(sequence,parent)
        if abs(len(sequence)-len(parent))>3 or not 1<=distance<=min(int(.3*len(parent)),6):return False,'local_edit_constraint'
    return passes_guardrails(sequence,reference_sequences=references)


@contextmanager
def destinations(paths,resume=False,inputs=()):
    paths=[Path(path).resolve() for path in paths]
    if len(set(paths))!=len(paths) or set(paths)&{Path(p).resolve() for p in inputs}:raise ValueError('Output paths overlap each other or inputs')
    handles=[]
    try:
        for path in sorted(paths):
            path.parent.mkdir(parents=True,exist_ok=True)
            handle=path.with_name(path.name+'.lock').open('a');handles.append(handle)
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise ValueError('Generation destination locked') from exc
        if not resume and any(path.exists() for path in paths):raise FileExistsError('Generation output exists; choose new output or explicit resume')
        yield
    finally:
        for handle in handles:handle.close()


def atomic_text(path,text):
    path=Path(path);fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def rng_state(device):
    result={'torch_rng':torch.get_rng_state(),'python_rng':random.getstate()}
    if device.startswith('cuda'):result['cuda_rng']=torch.cuda.get_rng_state_all()
    if device.startswith('mps'):result['mps_rng']=torch.mps.get_rng_state()
    return result


def restore_rng(state,device):
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    if device.startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    if device.startswith('mps'):torch.mps.set_rng_state(state['mps_rng'])


def sample(args,model,checkpoint):
    hashes=input_hashes(args);references=eligibility_context(args,required=bool(args.eligible_output))
    if args.count<1:raise ValueError('count must be positive')
    manifest_path=args.output+'.manifest.json'
    paths=[args.output,manifest_path]+([args.eligible_output] if args.eligible_output else [])
    with destinations(paths,inputs=[getattr(args,key) for key in hashes]):
        with torch.no_grad():sequences=[model.sample(args.parent,args.top_p,args.temperature) for _ in range(args.count)]
        rows=[];seen=set()
        for i,sequence in enumerate(sequences):
            eligible,reason=assess(sequence,args.parent,references) if references is not None else (False,'references_not_provided_not_assessed')
            duplicate=sequence in seen;seen.add(sequence)
            rows.append({'candidate_id':f'proposal_{i:05}','sequence':sequence,'eligible':eligible,'eligibility_reason':reason,'duplicate_proposal':duplicate,'candidate_pool_member':eligible and not duplicate})
        frame=pd.DataFrame(rows);verify_inputs(args,hashes)
        atomic_text(args.output,frame.to_csv(index=False))
        if args.eligible_output:atomic_text(args.eligible_output,frame[frame.candidate_pool_member].to_csv(index=False))
        manifest={'status':'complete','arguments':vars(args),'input_sha256':hashes,'legal_support':SUPPORT,'policy_stage':checkpoint.get('metadata',{}).get('stage','legacy_supervised_or_unknown'),'sampled_proposals':len(frame),'eligible_unique':int(frame.candidate_pool_member.sum()),'eligibility_assessed':references is not None,'paper_proposal_settings':args.count==200 and args.top_p==.95 and args.temperature==1.,'output_sha256':{str(path):sha256_file(path) for path in paths if str(path)!=manifest_path}}
        atomic_text(manifest_path,json.dumps(manifest,indent=2))


def reinforce(args,model,checkpoint,load):
    from src.generation.cli import atomic_torch_save,write_training_log
    if args.learning_rate<=0 or args.weight_decay<0:raise ValueError('Invalid optimizer settings')
    hashes=input_hashes(args);reference,ref=load(args.reference,args.device)
    if ref['conditional']!=checkpoint['conditional'] or ref['config']!=checkpoint['config']:raise ValueError('Reference task or architecture differs')
    identity=reference_identity(ref)
    if checkpoint.get('metadata',{}).get('stage')=='reinforce':
        if checkpoint['metadata'].get('reference_sha256')!=hashes['reference']:raise ValueError('Current RL policy belongs to another SFT reference')
    elif hashes['checkpoint']!=hashes['reference']:raise ValueError('Initial policy must be the selected SFT reference or a verifiable RL descendant')
    reference.eval().requires_grad_(False);model.eval() # identical deterministic trajectory/logprob distributions (dropout disabled).
    references=eligibility_context(args,required=True)
    state_path=args.training_state or str(Path(args.output).with_suffix('.rl.training.pt'))
    log_path=args.training_log or args.output+'.jsonl';failure_path=args.output+'.failure.json'
    paths=[args.output,state_path,log_path,failure_path]
    binding={'arguments':{k:v for k,v in vars(args).items() if k not in ('steps','resume_training')},'input_sha256':hashes,'implementation':{name:sha256_file(Path(__file__).with_name(name)) for name in ('runtime.py','policy.py','cli.py')},'runtime':{'python':platform.python_version(),'torch':str(torch.__version__)},'legal_support':SUPPORT}
    with destinations(paths,args.resume_training,inputs=[getattr(args,key) for key in hashes]):
        module,function=args.scorer.split(':');scorer=getattr(importlib.import_module(module),function)(args.scorer_config)
        # Official scorer validation occurs even if every initial trajectory is ineligible.
        if args.scorer=='src.score.frozen_scorer:FrozenSequenceScorer':scorer._initialize()
        binding['scorer_artifact_hashes']=getattr(scorer,'hashes',{})
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
        first=0;history=[];published_hash=None
        if args.resume_training:
            state=torch.load(state_path,map_location='cpu',weights_only=True)
            if state.get('binding')!=binding:raise ValueError('RL resume input/configuration mismatch')
            if Path(args.output).exists() and sha256_file(args.output)!=state.get('output_sha256'):raise ValueError('RL output hash mismatch')
            published_hash=state.get('output_sha256')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['next_step'];history=state['history'];restore_rng(state,args.device)
            if args.steps<first:raise ValueError('Requested steps precede saved state')
        state={'schema_version':1,'binding':binding,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'next_step':first,'history':history,'output_sha256':published_hash,**rng_state(args.device)}
        if not args.resume_training:atomic_torch_save(state,state_path)
        write_training_log(log_path,history)
        trajectories=[]
        try:
            for step in range(first,args.steps):
                with torch.no_grad():sequences=[model.sample(args.parent,1.,1.) for _ in range(args.batch_size)]
                trajectories=[]
                for sequence in sequences:
                    ok,reason=assess(sequence,args.parent,references)
                    trajectories.append({'sequence':sequence,'eligible':ok,'eligibility_reason':reason,'reward':0.,'scoring_status':'pending' if ok else 'ineligible'})
                selected=[row['sequence'] for row in trajectories if row['eligible']]
                if selected:
                    try:
                        values=list(scorer(selected))
                        if len(values)!=len(selected) or any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in values):raise ValueError('Invalid frozen R_t')
                    except Exception:
                        for row in trajectories:
                            if row['eligible']:row['scoring_status']='scoring_failure'
                        raise
                    it=iter(values)
                    for row in trajectories:
                        if row['eligible']:row.update(reward=float(next(it)),scoring_status='complete')
                loss=reinforce_loss(model,reference,sequences,[r['reward'] for r in trajectories],args.beta_kl,args.parent)
                if not torch.isfinite(loss):raise ValueError('Nonfinite RL loss')
                optimizer.zero_grad();loss.backward();optimizer.step()
                verify_inputs(args,hashes)
                if hasattr(scorer,'verify_artifacts'):scorer.verify_artifacts()
                history.append({'step':step+1,'loss':float(loss.detach()),'eligible_count':len(selected),'mean_reward':sum(r['reward'] for r in trajectories)/len(trajectories),'trajectories':trajectories})
                state=dict(schema_version=1,binding=binding,model=model.state_dict(),optimizer=optimizer.state_dict(),next_step=step+1,history=history,output_sha256=published_hash,**rng_state(args.device))
                atomic_torch_save(state,state_path);write_training_log(log_path,history)
                print(f'RL step={step+1}/{args.steps} eligible={len(selected)}/{len(sequences)} loss={float(loss.detach()):.6f}',flush=True)
            if any(sha256_file(Path(__file__).with_name(name))!=digest for name,digest in binding['implementation'].items()):raise ValueError('Generation implementation changed during RL')
            metadata={'stage':'reinforce','legal_support':SUPPORT,'reference_identity':identity,'reference_sha256':hashes['reference'],'arguments':vars(args),'input_hashes':hashes,'scorer_artifact_hashes':binding['scorer_artifact_hashes'],'steps':args.steps}
            verify_inputs(args,hashes)
            if hasattr(scorer,'verify_artifacts'):scorer.verify_artifacts()
            atomic_torch_save(dict(config=model.config,state_dict=model.state_dict(),conditional=checkpoint['conditional'],metadata=metadata),args.output)
            state['output_sha256']=sha256_file(args.output);atomic_torch_save(state,state_path)
            if Path(failure_path).exists():Path(failure_path).unlink()
        except BaseException as exc:
            atomic_text(failure_path,json.dumps({'status':'failed','error':str(exc),'completed_steps':state['next_step'],'failed_trajectories':trajectories,'resume_state':str(state_path)},indent=2));raise
