"""Iterative MIC-head calibration from standardized experimental observations."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from mdramp.calibration_train import CalibrationMLP, fresh_output, interval_huber
from mdramp.input_formats import read_experiments, aggregate_best

FROZEN_KEYS = ('esm_checkpoint','scaler','amp_prior','membrane','target_affinity',
               'target_embeddings','target_embedding_metadata','panel_manifest')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_head(x, best, starting_model, args):
    """Fit shared weights; intervals and labels never become inference overrides."""
    torch.manual_seed(args.seed)
    values=torch.as_tensor(x,dtype=torch.float32,device='cpu')
    lo=torch.as_tensor(best.lower_log2.to_numpy(dtype=float,copy=True),dtype=torch.float32)
    hi=torch.as_tensor(best.upper_log2.to_numpy(dtype=float,copy=True),dtype=torch.float32)
    if values.ndim != 2 or values.shape != (len(best),1571) or not torch.isfinite(values).all():
        raise ValueError('Expected finite N x 1571 frozen feature matrix')
    if torch.isnan(lo).any() or torch.isnan(hi).any() or (lo>hi).any():
        raise ValueError('Invalid MIC intervals')
    model=copy.deepcopy(starting_model).cpu() if starting_model is not None else CalibrationMLP(1571,16)
    model.requires_grad_(True)
    if starting_model is None:
        with torch.no_grad():
            model.layers[-1].weight.zero_()
            model.layers[-1].bias.fill_(float(torch.median(lo)))
    model.eval()
    with torch.no_grad():
        before=model(values).squeeze(-1).numpy().copy() if starting_model is not None else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(args.seed)
    history=[]
    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(values),generator=generator).split(args.batch_size):
            optimizer.zero_grad()
            loss=interval_huber(model(values[batch]).squeeze(-1),lo[batch],hi[batch])
            if not torch.isfinite(loss):raise ValueError('Nonfinite calibration loss')
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():loss=float(interval_huber(model(values).squeeze(-1),lo,hi))
        history.append({'epoch':epoch+1,'interval_huber':loss})
    with torch.no_grad():after=model(values).squeeze(-1).numpy()
    if not np.isfinite(after).all():raise ValueError('Nonfinite calibrated predictions')
    return model,before,after,history


def run(args):
    from mdramp.release_assets import resolve_model, promote_final
    from mdramp.release_runtime import load_runtime
    if min(args.epochs,args.batch_size,args.threads)<1 or args.learning_rate<=0 or not math.isfinite(args.learning_rate):
        raise ValueError('Positive finite learning rate, epochs, batch size and threads required')
    if not math.isfinite(args.mic_divisor) or args.mic_divisor<=0:
        raise ValueError('MIC divisor must be positive and finite')
    destination=Path(args.output_dir).resolve()
    if destination.exists():raise FileExistsError(f'Output already exists: {destination}')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    observations=read_experiments(args.input)
    best=aggregate_best(observations,divisor=args.mic_divisor)
    source_hash=sha(args.input)
    bundle=resolve_model(args.model)
    runtime=load_runtime(bundle)
    x,evidence=runtime.extract(best.sequence.tolist())
    if 'sequence' in evidence and evidence.sequence.tolist()!=best.sequence.tolist():
        raise ValueError('Expert feature row order differs from calibration sequences')
    frozen={k:runtime.hashes[k] for k in FROZEN_KEYS}
    parent_path=bundle.get('checkpoint')
    parent_hash=sha(parent_path) if parent_path else sha(bundle['scorer_config'])
    settings={k:getattr(args,k) for k in ('epochs','learning_rate','batch_size','threads','seed','mic_divisor')}
    identity=json.dumps({'parent':parent_hash,'input':source_hash,'settings':settings},sort_keys=True)
    state_id='calibrated_'+hashlib.sha256(identity.encode()).hexdigest()[:16]
    model,before,after,history=train_head(x,best,runtime.model,args)
    metadata=copy.deepcopy(runtime.checkpoint) if runtime.checkpoint else {}
    metadata.update(task='preference_mic_head',state_id=state_id,input_dim=1571,hidden_dim=16,
        output_dim=1,activation='GELU',model_kind='prospective_1571',fold='all',
        target_semantics='explicit_training_preference_not_measured_mic',target_divisor=None,
        calibration_mic_divisor=args.mic_divisor,input_target_semantics='user_supplied_original_scale_mic',
        historical_reproduction=False,inference_sequence_lookup=False,inference_rank_override=False,
        mean=[0.]*1571,scale=[1.]*1571,frozen_artifact_sha256=frozen,
        baseline_checkpoint_sha256=parent_hash,training_input_sha256={'experiments':source_hash},
        parent_model=str(args.model),parent_checkpoint_sha256=parent_hash,
        supervision='Interval Huber on cross-strain best MIC log2 bounds minus log2(calibration_mic_divisor)',
        training_config=settings,calibration_sequence_count=len(best),validation_performed=False,
        biological_improvement_claim=False,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})
    # A previous preference-fit selection is not evidence about this new calibration.
    for key in ('selection','requested_rank_count','fitting_sequence_count','train_peptide_ids'):
        metadata.pop(key,None)
    with fresh_output(destination) as out:
        observations.to_csv(out/'observations.csv',index=False)
        best.to_csv(out/'training_targets.csv',index=False)
        shutil.copy2(args.input,out/('experiments_original'+Path(args.input).suffix.lower()))
        torch.save(metadata,out/'model.pt')
        restored=CalibrationMLP(1571,16)
        restored.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
        restored.eval()
        with torch.no_grad():reloaded=restored(torch.as_tensor(x,dtype=torch.float32)).squeeze(-1).numpy()
        if not np.array_equal(after,reloaded):raise ValueError('Saved model predictions differ after reload')
        result=best[['id','sequence']].copy()
        result['before_mic']=np.exp2(before.astype(float)) if before is not None else np.nan
        result['after_mic']=np.exp2(reloaded.astype(float))
        if not np.isfinite(result.after_mic).all():raise ValueError('Nonfinite MIC after exponentiation')
        result.to_csv(out/'calibration.csv',index=False)
        pd.DataFrame(history).to_csv(out/'training_log.csv',index=False)
        config={k:os.path.relpath(Path(runtime.config[k]).absolute() if k=='esm_checkpoint' else Path(runtime.config[k]).resolve(),destination) for k in FROZEN_KEYS}
        config.update(mic_checkpoint='model.pt',scorer_kind='preference_mic_head',device='cpu',
                      generator=os.path.relpath(Path(bundle['generator']).absolute(),destination),
                      pair_batch_size=runtime.config.get('pair_batch_size',128))
        (out/'scorer.json').write_text(json.dumps(config,indent=2))
        if sha(args.input)!=source_hash:raise ValueError('Experimental input changed during calibration')
        for key,digest in frozen.items():
            if sha(runtime.config[key])!=digest:raise ValueError('Frozen asset changed during calibration: '+key)
        record={'status':'complete','state_id':state_id,'task':'iterative_calibration','starting_model':str(args.model),
                'parent_checkpoint_sha256':parent_hash,'source_sha256':source_hash,'mic_unit':'μg/mL',
                'unique_sequences':len(best),'source_observations':len(observations),'settings':settings,
                'initialized_mic_head':runtime.model is None,'saved_model_reload_equal':True,
                'frozen_artifact_sha256':frozen,'implementation_sha256':sha(__file__),
                'prediction_semantics':'user-adjusted best-MIC prediction; not experimental confirmation',
                'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
        (out/'manifest.json').write_text(json.dumps(record,indent=2))
    if args.update_final:promote_final(destination/'scorer.json')
    print(f'Calibrated {len(best)} peptides. Results: {destination}')
    return destination
