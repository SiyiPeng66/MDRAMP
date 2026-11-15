"""Train and reload an explicit preference-conditioned MIC head.

Labels and requested ranks are training constraints only. Inference applies one
shared 1571->16->1 neural network to every feature row, without sequence lookup,
per-sequence scaling, rank overrides, or access to training targets.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mdramp.calibration_train import CalibrationMLP, fresh_output, sha, _json

TASK = 'preference_mic_head'


def verify_design_manifest(design_path, table_path, checkpoint, baseline_hash):
    """Bind cached features to their frozen panel/experts and warm-start model."""
    manifest_path=Path(design_path).resolve().parent/'manifest.json'
    if not manifest_path.exists():return None
    manifest=json.loads(manifest_path.read_text())
    outputs=manifest.get('output_sha256',{})
    for path in [Path(design_path),Path(table_path)]:
        if outputs.get(path.name)!=sha(path):
            raise ValueError('Design manifest output checksum mismatch: '+path.name)
    if manifest.get('frozen_artifact_sha256')!=checkpoint.get('frozen_artifact_sha256'):
        raise ValueError('Design and checkpoint frozen artifacts differ')
    if baseline_hash not in manifest.get('input_sha256',{}).values():
        raise ValueError('Design manifest does not bind the baseline checkpoint')
    if manifest.get('target_semantics')!='explicit_user_preference_targets_not_measured_MIC':
        raise ValueError('Design manifest must declare explicit preference targets')
    return manifest_path


def boolean_column(values):
    normalized=values.astype(str).str.lower()
    if not normalized.isin(['true','false','1','0']).all():
        raise ValueError('Boolean group membership required')
    return normalized.isin(['true','1']).to_numpy()


def load_design(path, table, training=False):
    frame=pd.read_csv(table)
    required={'sequence','round','p_cons','group_in_600'}
    if training:required|={'baseline_log2','target_log2','requested_rank'}
    if required-set(frame):raise ValueError('Missing design columns: '+str(sorted(required-set(frame))))
    if frame.sequence.isna().any() or frame.sequence.duplicated().any():raise ValueError('Unique nonmissing sequences required')
    with np.load(path,allow_pickle=False) as z:
        x=np.asarray(z['x'],dtype=np.float32)
        if z['sequences'].astype(str).tolist()!=frame.sequence.tolist():raise ValueError('NPZ/CSV sequence order mismatch')
    if x.shape!=(len(frame),1571) or not np.isfinite(x).all():raise ValueError('Finite N x 1571 design required')
    if not np.isfinite(frame.p_cons).all() or not frame.p_cons.between(0,1).all():raise ValueError('Finite p_cons in [0,1] required')
    grouped=boolean_column(frame.group_in_600)
    if frame.loc[grouped,'round'].isna().any():raise ValueError('Ranked candidate groups need a round')
    if training:
        if not np.isfinite(frame[['baseline_log2','target_log2']].to_numpy(float)).all():raise ValueError('Finite training targets required')
        request=frame.requested_rank.notna()
        if not request.any():raise ValueError('At least one requested rank required')
        if (request & ~grouped).any():raise ValueError('Requested ranks must belong to candidate groups')
        for r in frame.loc[request].itertuples():
            n=int(((frame['round']==r.round)&grouped).sum())
            if not float(r.requested_rank).is_integer() or not 1<=r.requested_rank<=n:raise ValueError('Requested rank outside group')
    return frame,torch.from_numpy(x)


def score(model,x,frame):
    with torch.inference_mode():pred=model(x).squeeze(-1).cpu().numpy()
    result=frame.drop(columns=['target_log2','requested_rank'],errors='ignore').copy()
    result['predicted_preference_log2_mic']=pred
    result['predicted_preference_mic_ug_ml']=np.exp2(pred.astype(float))
    result['A_t']=1/(1+np.exp(pred.astype(float)-5))
    result['R_t']=result.p_cons*result.A_t
    if not np.isfinite(result[['predicted_preference_mic_ug_ml','R_t']].to_numpy()).all():raise ValueError('Nonfinite inference output')
    result['rank']=pd.Series(pd.NA,index=result.index,dtype='Int64')
    grouped=boolean_column(result.group_in_600)
    for _,g in result[grouped].groupby('round'):
        indices=g.sort_values(['R_t','sequence'],ascending=[False,True]).index
        result.loc[indices,'rank']=np.arange(1,len(indices)+1)
    return result


def rank_error(result,frame):
    request=frame.requested_rank.notna()
    return int(np.abs(result.loc[request,'rank'].astype(int).to_numpy()-frame.loc[request,'requested_rank'].to_numpy(int)).sum())


def requested_rank_loss(pred,p_cons,constraints,margin):
    """Fit target MIC to the current neighboring-score gap (training only).

    Other candidates define detached boundaries; all forward values remain a
    shared neural prediction. Inverting score boundaries to log2 MIC avoids
    gradients vanishing when the activity sigmoid is nearly saturated.
    """
    scores=(p_cons*torch.sigmoid(5-pred)).detach()
    penalties=[]
    for index,others,rank in constraints:
        ordered=torch.sort(scores[others],descending=True).values
        if rank>1:
            upper=ordered[rank-2]-margin
            activity=(upper/p_cons[index]).clamp(1e-7,1-1e-7)
            lower_mic=5+torch.log(1/activity-1)
            penalties.append(torch.relu(lower_mic-pred[index]).square())
        if rank<=len(ordered):
            lower=ordered[rank-1]+margin
            activity=(lower/p_cons[index]).clamp(1e-7,1-1e-7)
            upper_mic=5+torch.log(1/activity-1)
            penalties.append(torch.relu(pred[index]-upper_mic).square())
    return torch.stack(penalties).mean() if penalties else pred.sum()*0


def model_from_checkpoint(checkpoint):
    if checkpoint.get('task')!=TASK or checkpoint.get('input_dim')!=1571 or checkpoint.get('hidden_dim')!=16:
        raise ValueError('Expected preference MIC head checkpoint')
    model=CalibrationMLP(1571,16)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model


def train(args):
    frame,x=load_design(args.design,args.table,training=True)
    baseline=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if baseline.get('task')!='prospective_adjusted_best_mic' or baseline.get('input_dim')!=1571 or baseline.get('hidden_dim')!=16:
        raise ValueError('Warm start requires baseline prospective 1571->16->1 model')
    design_manifest=verify_design_manifest(args.design,args.table,baseline,sha(args.checkpoint))
    model=CalibrationMLP(1571,16);model.load_state_dict(baseline['state_dict']);model.eval()
    with torch.no_grad():base=model(x).squeeze(-1).clone()
    if not np.allclose(base.numpy(),frame.baseline_log2.to_numpy(float),atol=2e-4,rtol=1e-5):
        raise ValueError('Baseline predictions do not match supplied features/checkpoint')
    targets=torch.tensor(frame.target_log2.to_numpy(),dtype=torch.float32)
    request=torch.tensor(frame.requested_rank.notna().to_numpy())
    others=~request
    p_cons=torch.tensor(frame.p_cons.to_numpy(),dtype=torch.float32)
    grouped=boolean_column(frame.group_in_600)
    constraints=[]
    for index,row in frame[frame.requested_rank.notna()].iterrows():
        neighbors=np.flatnonzero((frame['round']==row['round']).to_numpy() & grouped & (np.arange(len(frame))!=index))
        constraints.append((int(index),torch.tensor(neighbors),int(row.requested_rank)))
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    paths={k:str(Path(getattr(args,k)).resolve()) for k in ['design','table','checkpoint']}
    if design_manifest:paths['design_manifest']=str(design_manifest)
    hashes={k:sha(v) for k,v in paths.items()}
    best=None;best_key=None;history=[]
    for epoch in range(args.epochs+1):
        if epoch:
            model.train();optimizer.zero_grad();pred=model(x).squeeze(-1)
            target_loss=(pred-targets).square().mean()
            request_loss=(pred[request]-targets[request]).square().mean()
            retention_loss=(pred[others]-base[others]).square().mean() if others.any() else pred.sum()*0
            rank_loss=requested_rank_loss(pred,p_cons,constraints,args.rank_margin)
            loss=target_loss+args.target_weight*request_loss+args.retention*retention_loss+args.rank_weight*rank_loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step()
        if epoch%args.check_every and epoch!=args.epochs:continue
        model.eval();result=score(model,x,frame)
        pred=result.predicted_preference_log2_mic.to_numpy()
        err=rank_error(result,frame)
        drift=float(np.mean((pred[others.numpy()]-base.numpy()[others.numpy()])**2)) if others.any() else 0.
        fit=float(np.mean((pred-targets.numpy())**2))
        key=(err,drift,fit)
        history.append({'epoch':epoch,'rank_error':err,'nonrequested_baseline_mse':drift,'target_mse':fit})
        if best_key is None or key<best_key:
            best_key=key;best={'state_dict':copy.deepcopy(model.state_dict()),'epoch':epoch,'rank_error':err,'nonrequested_baseline_mse':drift,'target_mse':fit}
        if epoch%500==0:print(f'epoch={epoch} rank_error={err} drift={drift:.6g} best_rank_error={best_key[0]}',flush=True)
    checkpoint=copy.deepcopy(baseline)
    checkpoint['baseline_train_peptide_ids']=checkpoint.pop('train_peptide_ids',[])
    checkpoint.update(task=TASK,state_id=args.state_id,state_dict=best.pop('state_dict'),
        target_semantics='explicit_training_preference_not_measured_mic',target_divisor=None,
        baseline_checkpoint_sha256=hashes['checkpoint'],training_input_sha256=hashes,
        fitting_sequence_count=len(frame),requested_rank_count=int(request.sum()),
        training_config=vars(args),selection=best,
        supervision='Shared neural MIC head warm-started on explicit preference targets; checkpoint selected by requested training ranks then nonrequested baseline drift',
        historical_reproduction=False,validation_performed=False,
        inference_sequence_lookup=False,inference_rank_override=False)
    with fresh_output(args.output_dir) as out:
        torch.save(checkpoint,out/'model.pt')
        # Mandatory disk reload: published values come from saved weights.
        restored=model_from_checkpoint(torch.load(out/'model.pt',map_location='cpu',weights_only=True))
        result=score(restored,x,frame);result['state_id']=args.state_id
        result['prediction_semantics']=checkpoint['target_semantics']
        result.to_csv(out/'scores.csv',index=False)
        pd.DataFrame(history).to_csv(out/'training_log.csv',index=False)
        requested=frame.loc[frame.requested_rank.notna(),['sequence','round','target_log2','requested_rank']].merge(result[['sequence','predicted_preference_mic_ug_ml','R_t','rank']],on='sequence',validate='one_to_one')
        requested.to_csv(out/'requested_rank_fit.csv',index=False)
        if any(sha(paths[k])!=v for k,v in hashes.items()):raise ValueError('Inputs changed during training')
        _json(out/'manifest.json',{'task':TASK,'state_id':args.state_id,'status':'complete','requested_ranks_achieved':rank_error(result,frame)==0,'selection':best,'input_sha256':hashes,'implementation_sha256':sha(__file__),'config':vars(args),'validation_performed':False,'interpretation':'Training preference fit on a fixed candidate set; not independent validation or measured MIC','output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})


def predict(args):
    frame,x=load_design(args.design,args.table)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    verify_design_manifest(args.design,args.table,checkpoint,checkpoint.get('baseline_checkpoint_sha256'))
    model=model_from_checkpoint(checkpoint)
    result=score(model,x,frame);result['state_id']=checkpoint['state_id'];result['prediction_semantics']=checkpoint['target_semantics']
    with fresh_output(args.output_dir) as out:
        result.to_csv(out/'scores.csv',index=False)
        _json(out/'manifest.json',{'task':'preference_head_inference','state_id':checkpoint['state_id'],'training_targets_used':False,'sequence_lookup_used':False,'input_sha256':{k:sha(getattr(args,k)) for k in ['design','table','checkpoint']},'output_sha256':{'scores.csv':sha(out/'scores.csv')}})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','predict'])
    for name in ['design','table','checkpoint','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--state-id',default='prospective575_preference_head_v1')
    p.add_argument('--epochs',type=int,default=3000);p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--target-weight',type=float,default=100.);p.add_argument('--retention',type=float,default=1.)
    p.add_argument('--rank-weight',type=float,default=1000.);p.add_argument('--rank-margin',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--threads',type=int,default=2);p.add_argument('--check-every',type=int,default=10)
    a=p.parse_args(argv)
    if min(a.epochs,a.threads,a.check_every)<1 or not np.isfinite([a.learning_rate,a.target_weight,a.retention,a.rank_weight,a.rank_margin]).all() or a.learning_rate<=0 or min(a.target_weight,a.retention,a.rank_weight,a.rank_margin)<0:raise ValueError('Invalid hyperparameters')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);torch.use_deterministic_algorithms(True)
    (train if a.action=='train' else predict)(a)

if __name__=='__main__':main()
