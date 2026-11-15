"""Retrospective peptide-best-panel censored MIC calibration; not a paper-prequential model.

Fixed five-fold clustered OOF comparison; no test-fold tuning or early stopping.
Threshold evidence uses fixed-temperature sigmoid scores, not calibrated probabilities.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn

# Compatibility label for one pooled peptide-level output, not a strain identity.
STRAINS = ('best_panel',)
TASK = 'peptide_best_mic'
LIMITATIONS = [
    'Retrospective extension MIC calibration head; does not replace manuscript historical MIC training.',
    'Clustered OOF assessment is not paper-prequential or prospective validation.',
    'ESM3+physicochemical final architecture is prespecified; OOF metrics do not select architecture or hyperparameters.',
    'Threshold sigmoid scores use fixed temperature 1 log2 MIC unit and are not calibrated probabilities.',
    'MIC units are ug/mL as documented in the source workbook; no concentration-unit conversion is performed.',
    'The output is the lowest MIC across the declared panel, not broad-spectrum or per-strain potency.',
    'The ESM3+32-descriptor head omits the three expert means: full manuscript 1571-input integration is not completed; no integrated Rt is produced.',
    'Censored observations constrain intervals; qualitative +/- readings are never converted into MIC values.',
]


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


@contextmanager
def fresh_output(destination):
    destination=Path(destination).absolute();destination.parent.mkdir(parents=True,exist_ok=True)
    with (destination.parent/('.'+destination.name+'.lock')).open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Calibration output is locked') from exc
        if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
        temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
        try:
            yield temporary
            if destination.exists():raise FileExistsError(destination)
            os.rename(temporary,destination)
        finally:
            if temporary.exists():shutil.rmtree(temporary)


def _read(path,required):
    frame=pd.read_csv(path,dtype={'peptide_id':str,'record_id':str,'sequence':str,'cluster_id':str,'strain_id':str})
    if not set(required).issubset(frame):raise ValueError(f'Missing columns in {path}: {set(required)-set(frame)}')
    if frame[list(required)].isna().any().any():raise ValueError(f'Missing values in {path}')
    return frame


def load_design(features,embeddings,metadata,sequences=None):
    features=_read(features,['peptide_id','sequence'])
    if features.peptide_id.duplicated().any():raise ValueError('Duplicate feature peptide_id')
    columns=[c for c in features if c.startswith('pc_')]
    if len(columns)!=32:raise ValueError('Exactly 32 physicochemical descriptors required')
    if sequences is None:sequences=features[['peptide_id','sequence']].copy()
    lookup=features.set_index('peptide_id')
    if not set(sequences.peptide_id).issubset(lookup.index):raise ValueError('Missing feature IDs')
    aligned=lookup.loc[sequences.peptide_id]
    if aligned.sequence.tolist()!=sequences.sequence.tolist():raise ValueError('Feature sequence mismatch')
    meta=_read(metadata,['record_id','sequence','global_index'])
    if meta.record_id.duplicated().any() or meta.global_index.duplicated().any():raise ValueError('Duplicate embedding IDs/indices')
    with np.load(embeddings,allow_pickle=False) as archive:vectors=np.asarray(archive['global_embeddings'],dtype=np.float32)
    index=pd.to_numeric(meta.global_index,errors='raise').to_numpy()
    if vectors.ndim!=2 or not np.isfinite(index).all() or not np.equal(index,index.astype(int)).all() or (index<0).any() or (index>=len(vectors)).any():raise ValueError('Invalid embedding shape/index')
    meta=meta.set_index('record_id')
    if not set(sequences.peptide_id).issubset(meta.index):raise ValueError('Missing embedding IDs')
    selected=meta.loc[sequences.peptide_id]
    if selected.sequence.tolist()!=sequences.sequence.tolist():raise ValueError('Embedding sequence mismatch')
    pc=aligned[columns].to_numpy(dtype=np.float32);esm=vectors[selected.global_index.to_numpy(dtype=int)]
    if not np.isfinite(pc).all() or not np.isfinite(esm).all():raise ValueError('Nonfinite features/embeddings')
    return sequences.reset_index(drop=True),pc,np.concatenate([esm,pc],axis=1),columns


def target_metadata(frame):
    """Reject mixed measurement/derived-target semantics before fitting any model."""
    present={'target_semantics','target_divisor'} & set(frame)
    preserved_measurements=any(str(column).startswith('measured_') for column in frame)
    if not present:
        if preserved_measurements:raise ValueError('Preserved measured columns require explicit design target semantics')
        return {'target_semantics':'measured_mic','target_divisor':1.}
    if present!={'target_semantics','target_divisor'} or frame.empty:
        raise ValueError('Both target_semantics and target_divisor must be explicit')
    if frame.target_semantics.isna().any() or frame.target_semantics.nunique()!=1:
        raise ValueError('Mixed or missing target semantics')
    semantics=str(frame.target_semantics.iloc[0])
    divisors=pd.to_numeric(frame.target_divisor,errors='raise').to_numpy(float)
    if not np.isfinite(divisors).all() or len(set(divisors))!=1:
        raise ValueError('Mixed or nonfinite target divisors')
    divisor=float(divisors[0])
    if semantics=='measured_mic':
        if preserved_measurements:raise ValueError('Preserved measured columns conflict with measured target semantics')
        if divisor!=1.:raise ValueError('Measured MIC must have target_divisor=1')
    elif semantics=='user_assigned_screen_mic':
        if divisor!=1.:raise ValueError('User-assigned screening MIC must remain on original scale')
        if not {'source_interval_lower_log2','source_interval_upper_log2'}.issubset(frame):
            raise ValueError('Assigned screening targets require preserved source intervals')
    elif semantics=='design_target_not_measured':
        if divisor<=1.:raise ValueError('Design target divisor must exceed one')
        required={'measured_lower_log2','measured_upper_log2','measured_exact_log2'}
        if not required.issubset(frame):raise ValueError('Design targets require preserved measured bounds and exact values')
        usable=frame[frame.status.eq('usable')] if 'status' in frame else frame
        shift=np.log2(divisor)
        for field in ('lower_log2','upper_log2'):
            measured=pd.to_numeric(usable['measured_'+field],errors='raise').to_numpy(float)
            derived=pd.to_numeric(usable[field],errors='raise').to_numpy(float)
            if not np.allclose(derived,measured-shift,rtol=0,atol=1e-10,equal_nan=False):
                raise ValueError('Design target bounds do not match measured bounds / divisor')
        measured_exact=pd.to_numeric(usable.measured_exact_log2,errors='raise').to_numpy(float)
        original_lower=pd.to_numeric(usable.measured_lower_log2,errors='raise').to_numpy(float)
        original_upper=pd.to_numeric(usable.measured_upper_log2,errors='raise').to_numpy(float)
        exact=np.isfinite(original_lower)&(original_lower==original_upper)
        if not np.allclose(measured_exact[exact],original_lower[exact],rtol=0,atol=1e-10) or not np.isnan(measured_exact[~exact]).all():
            raise ValueError('Preserved measured exact values contradict measured intervals')
        if 'exact_log2' in usable:
            derived_exact=pd.to_numeric(usable.exact_log2,errors='raise').to_numpy(float)
            if not np.allclose(derived_exact,measured_exact-shift,rtol=0,atol=1e-10,equal_nan=True):
                raise ValueError('Derived exact values do not match measured values / divisor')
    else:raise ValueError('Unknown target semantics')
    return {'target_semantics':semantics,'target_divisor':divisor}


def checkpoint_target_metadata(checkpoint):
    metadata={key:checkpoint.get(key,default) for key,default in [('target_semantics','measured_mic'),('target_divisor',1.)]}
    semantics=metadata['target_semantics'];divisor=metadata['target_divisor']
    if not isinstance(divisor,(int,float)) or not np.isfinite(divisor) or (semantics=='measured_mic' and divisor!=1) or (semantics=='design_target_not_measured' and divisor<=1) or semantics not in {'measured_mic','design_target_not_measured'}:
        raise ValueError('Invalid checkpoint target semantics')
    return metadata


def load_observations(sequence_path,observation_path):
    sequences=_read(sequence_path,['peptide_id','sequence','cluster_id','fold'])
    if sequences.empty or sequences.peptide_id.duplicated().any() or sequences.sequence.duplicated().any():raise ValueError('Sequences and peptide IDs must be unique and nonempty')
    folds=pd.to_numeric(sequences.fold,errors='raise')
    if set(folds)!={0,1,2,3,4}:raise ValueError('Exactly five folds numbered 0..4 required')
    sequences['fold']=folds.astype(int)
    if sequences.groupby('cluster_id').fold.nunique().max()!=1:raise ValueError('Cluster split across folds')
    obs=_read(observation_path,['peptide_id','strain_id'])
    target=target_metadata(obs)
    if obs.peptide_id.duplicated().any():raise ValueError('Peptide-best MIC requires exactly one observation row per peptide')
    if not obs.strain_id.eq('best_panel').all():raise ValueError('Peptide-best MIC requires strain_id=best_panel')
    for key,value in target.items():obs[key]=value
    if 'status' in obs:
        if not obs.status.isin(['usable','missing','quarantined']).all():raise ValueError('Unknown observation status')
        obs=obs[obs.status.eq('usable')].copy()
    if not {'lower_log2','upper_log2'}.issubset(obs):raise ValueError('Missing MIC bounds')
    for name in ('lower_inclusive','upper_inclusive'):
        if name not in obs:obs[name]=True
        values=obs[name].astype(str).str.lower()
        if not values.isin(['true','false','1','0']).all():raise ValueError('Invalid interval inclusivity')
        obs[name]=values.isin(['true','1'])
    if obs.empty or not set(obs.strain_id).issubset(STRAINS):raise ValueError('Unknown/empty strain observations')
    if not set(obs.peptide_id).issubset(sequences.peptide_id):raise ValueError('Observation peptide absent from sequences')
    if obs[['peptide_id','strain_id']].duplicated().any():raise ValueError('Duplicate peptide-strain observations; aggregate source readings before training')
    lower=pd.to_numeric(obs.lower_log2,errors='raise').to_numpy(float);upper=pd.to_numeric(obs.upper_log2,errors='raise').to_numpy(float)
    if np.isnan(lower).any() or np.isnan(upper).any() or (lower>upper).any() or np.isposinf(lower).any() or np.isneginf(upper).any() or (np.isneginf(lower)&np.isposinf(upper)).any():raise ValueError('Invalid or unconstrained MIC interval')
    if ((lower==upper)&(~obs.lower_inclusive.to_numpy()|~obs.upper_inclusive.to_numpy())).any():raise ValueError('Empty exact interval')
    obs['lower_log2']=lower;obs['upper_log2']=upper
    canonical=sequences.set_index('peptide_id')
    for column in ('sequence','cluster_id','fold'):
        if column in obs:
            expected=canonical.loc[obs.peptide_id,column].to_numpy()
            actual=obs[column].to_numpy()
            if column=='fold':actual=pd.to_numeric(obs[column],errors='raise').to_numpy()
            else:actual=obs[column].astype(str).to_numpy();expected=expected.astype(str)
            if not np.equal(actual,expected).all():raise ValueError(f'Observation {column} contradicts sequence fold design')
            obs=obs.drop(columns=column)
    obs=obs.merge(sequences[['peptide_id','sequence','cluster_id','fold']],on='peptide_id',how='left',validate='many_to_one')
    obs['observation_index']=np.arange(len(obs))
    for fold in range(5):
        if set(obs.loc[obs.fold.ne(fold),'strain_id'])!=set(STRAINS):raise ValueError(f'Fold {fold} training observations have no best-panel target')
    return sequences,obs


def violation(pred,lower,upper):
    return np.maximum(np.maximum(lower-pred,pred-upper),0.)


def huber_values(pred,lower,upper):
    residual=violation(np.asarray(pred),np.asarray(lower),np.asarray(upper))
    return np.where(residual<=1.,.5*residual**2,residual-.5)


def interval_huber(pred,lower,upper):
    residual=torch.maximum(torch.maximum(lower-pred,pred-upper),torch.zeros_like(pred))
    return torch.where(residual<=1,.5*residual.square(),residual-.5).mean()


class CalibrationMLP(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__();self.layers=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
    def forward(self,x):return self.layers(x)


def fit_constant(obs):
    values=[]
    for strain in STRAINS:
        rows=obs[obs.strain_id.eq(strain)];lo=rows.lower_log2.to_numpy();hi=rows.upper_log2.to_numpy()
        finite=np.concatenate([lo[np.isfinite(lo)],hi[np.isfinite(hi)]])
        left,right=float(finite.min()-1),float(finite.max()+1)
        result=minimize_scalar(lambda x:huber_values(x,lo,hi).mean(),bounds=(left,right),method='bounded')
        if not result.success:raise RuntimeError('Constant baseline optimization failed')
        values.append(float(result.x))
    return np.asarray(values)


def fit_model(design,sequences,obs,args,seed):
    torch.manual_seed(seed)
    train_ids=set(obs.peptide_id)
    mask=sequences.peptide_id.isin(train_ids).to_numpy()
    # Each unique training peptide contributes once to scaler fitting.
    mean=design[mask].mean(axis=0,dtype=np.float64);scale=design[mask].std(axis=0,dtype=np.float64);scale[scale<1e-12]=1.
    transformed=((design-mean)/scale).astype(np.float32)
    lookup={key:i for i,key in enumerate(sequences.peptide_id)}
    x=torch.as_tensor(transformed[[lookup[key] for key in obs.peptide_id]],device=args.device)
    strain=torch.tensor([STRAINS.index(value) for value in obs.strain_id],device=args.device)
    lower=torch.tensor(obs.lower_log2.to_numpy(),dtype=torch.float32,device=args.device);upper=torch.tensor(obs.upper_log2.to_numpy(),dtype=torch.float32,device=args.device)
    model=CalibrationMLP(design.shape[1],args.hidden_dim).to(args.device)
    # Prespecified training-only initialization avoids an arbitrary 1 ug/mL start.
    with torch.no_grad():
        model.layers[-1].weight.zero_()
        model.layers[-1].bias.copy_(torch.as_tensor(fit_constant(obs),dtype=torch.float32,device=args.device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    generator=torch.Generator().manual_seed(seed)
    history=[]
    for epoch in range(args.epochs):
        total=0.
        for indices in torch.randperm(len(obs),generator=generator).split(args.batch_size):
            indices=indices.to(args.device);optimizer.zero_grad()
            pred=model(x[indices]).gather(1,strain[indices,None]).squeeze(1)
            loss=interval_huber(pred,lower[indices],upper[indices])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
        history.append(total/len(obs))
    model.eval()
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'input_dim':design.shape[1],'hidden_dim':args.hidden_dim,'mean':mean.tolist(),'scale':scale.tolist(),'strain_ids':list(STRAINS),'train_peptide_ids':sorted(train_ids),'epochs':args.epochs,'seed':seed,'train_loss':history,'initialization':'zero final weights; training-fold best-panel interval-Huber constant bias','task':TASK,'output_dim':1,'activation':'GELU','descriptor_dim':32,'esm_dim':design.shape[1]-32 if design.shape[1]>32 else 0,'three_expert_means_included':False,'full_manuscript_architecture':False}
    return model,checkpoint


def infer(model,checkpoint,design,args):
    x=((design-np.asarray(checkpoint['mean']))/np.asarray(checkpoint['scale'])).astype(np.float32)
    with torch.inference_mode():
        pred=np.concatenate([model(torch.as_tensor(x[i:i+args.batch_size],device=args.device)).cpu().numpy() for i in range(0,len(x),args.batch_size)])
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite predictions')
    return pred


def metrics(rows,prediction):
    pred=rows[prediction].to_numpy();lo=rows.lower_log2.to_numpy();hi=rows.upper_log2.to_numpy();exact=np.isfinite(lo)&(lo==hi)
    report={'n':len(rows),'interval_huber':float(huber_values(pred,lo,hi).mean()),'mean_interval_violation_log2':float(violation(pred,lo,hi).mean()),'exact_n':int(exact.sum()),'exact_mae_log2':float(np.abs(pred[exact]-lo[exact]).mean()) if exact.any() else None,'thresholds':{}}
    for threshold in (32,256):
        boundary=np.log2(threshold);lower_inclusive=rows.lower_inclusive.to_numpy() if 'lower_inclusive' in rows else np.ones(len(rows),dtype=bool);known=(hi<=boundary)|(lo>boundary)|((lo==boundary)&~lower_inclusive);labels=(hi[known]<=boundary).astype(int);scores=expit(boundary-pred[known])
        report['thresholds'][str(threshold)]={'known_n':int(known.sum()),'positive_n':int(labels.sum()),'unknown_n':int((~known).sum()),'brier':float(np.mean((scores-labels)**2)) if known.any() else None,'roc_auc':float(roc_auc_score(labels,scores)) if len(set(labels))==2 else None,'average_precision':float(average_precision_score(labels,scores)) if len(set(labels))==2 else None,'temperature_log2':1.,'probability_calibrated':False}
    return report


def bootstrap(rows,replicates,seed):
    rng=np.random.default_rng(seed);clusters=sorted(rows.cluster_id.unique());groups=[np.flatnonzero(rows.cluster_id.to_numpy()==key) for key in clusters]
    losses={name:huber_values(rows[name],rows.lower_log2,rows.upper_log2) for name in ('constant','pc','esm_pc')}
    result={}
    for baseline in ('constant','pc'):
        delta=np.asarray(losses['esm_pc']-losses[baseline]);sums=np.asarray([delta[g].sum() for g in groups]);counts=np.asarray([len(g) for g in groups]);samples=[]
        for _ in range(replicates):
            selected=rng.integers(0,len(groups),len(groups));samples.append(float(sums[selected].sum()/counts[selected].sum()))
        result['esm_pc_minus_'+baseline]={'mean_interval_huber_difference':float(delta.mean()),'cluster_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),'replicates':replicates,'negative_favors_esm_pc':True,'unit':'peptide_cluster','cluster_count':len(clusters)}
    return result


def _identity(args,names):
    paths={name:str(Path(getattr(args,name)).absolute()) for name in names}
    paths['implementation']=str(Path(__file__).absolute())
    return paths,{name:sha(path) for name,path in paths.items()}


def _finish(output,paths,hashes,details):
    if {name:sha(path) for name,path in paths.items()}!=hashes:raise RuntimeError('Input or implementation changed during calibration')
    manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'inputs':paths,'input_sha256':hashes,'runtime':{name:importlib.metadata.version(name) for name in ('numpy','pandas','torch','scipy','scikit-learn')},'limitations':LIMITATIONS,'three_expert_means_included':False,'full_manuscript_architecture':False,**details}
    manifest['output_sha256']={str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    _json(output/'manifest.json',manifest)


def train(args):
    paths,hashes=_identity(args,['sequences','observations','features','embeddings','embedding_metadata'])
    sequences,obs=load_observations(args.sequences,args.observations)
    target=target_metadata(obs)
    if target['target_semantics']=='user_assigned_screen_mic':raise ValueError('Assigned screening labels require the explicit prospective training pipeline')
    raw_observations=pd.read_csv(args.observations)
    status_counts=raw_observations.status.value_counts().to_dict() if 'status' in raw_observations else {'provided_usable':len(raw_observations)}
    sequences,pc,esm_pc,columns=load_design(args.features,args.embeddings,args.embedding_metadata,sequences)
    output_rows=obs.copy()
    for name in ('constant','pc','esm_pc'):output_rows[name]=np.nan
    with fresh_output(args.output_dir) as output:
        for fold in range(5):
            training=obs[obs.fold.ne(fold)];test=obs[obs.fold.eq(fold)]
            if set(training.peptide_id)&set(test.peptide_id) or set(training.cluster_id)&set(test.cluster_id):raise ValueError('Train/test peptide or cluster leakage')
            constants=fit_constant(training);fold_dir=output/f'fold_{fold}';fold_dir.mkdir()
            _json(fold_dir/'constant.json',{'task':TASK,'output_dim':1,'strain_ids':list(STRAINS),'values':constants.tolist(),'train_peptide_ids':sorted(set(training.peptide_id)),**target})
            output_rows.loc[test.index,'constant']=[constants[STRAINS.index(s)] for s in test.strain_id]
            lookup={key:i for i,key in enumerate(sequences.peptide_id)}
            for name,design in [('pc',pc),('esm_pc',esm_pc)]:
                model,checkpoint=fit_model(design,sequences,training,args,args.seed+fold)
                checkpoint.update(feature_columns=columns,model_kind=name,fold=fold,input_sha256=hashes,**target)
                torch.save(checkpoint,fold_dir/(name+'.pt'))
                prediction=infer(model,checkpoint,design,args)
                output_rows.loc[test.index,name]=[prediction[lookup[key],STRAINS.index(strain)] for key,strain in zip(test.peptide_id,test.strain_id)]
            print(f'calibration fold {fold+1}/5 complete; train observations={len(training)} heldout={len(test)}',flush=True)
        if not np.isfinite(output_rows[['constant','pc','esm_pc']].to_numpy()).all():raise RuntimeError('Incomplete OOF predictions')
        output_rows.to_csv(output/'oof_predictions.csv',index=False)
        report={'task':TASK,'output_dim':1,**target,'target_interpretation':'Derived design targets are hypothetical goals, not measured MIC improvement.' if target['target_semantics']=='design_target_not_measured' else 'Observed MIC interval prediction.','models':{name:{'overall':metrics(output_rows,name)} for name in ('constant','pc','esm_pc')},'paired_cluster_bootstrap':bootstrap(output_rows,args.bootstrap_replicates,args.seed),'limitations':LIMITATIONS,'interpretation':'Differences are retrospective OOF evidence only; final architecture was prespecified, not selected from these results.'}
        _json(output/'metrics.json',report)
        model,checkpoint=fit_model(esm_pc,sequences,obs,args,args.seed)
        checkpoint.update(feature_columns=columns,model_kind='esm_pc',fold='all',input_sha256=hashes,limitations=LIMITATIONS,**target)
        torch.save(checkpoint,output/'final_model.pt')
        _finish(output,paths,hashes,{'task':TASK,'output_dim':1,**target,'mode':'train','status':'complete','config':vars(args),'strain_ids':list(STRAINS),'observations':len(obs),'input_observations':len(raw_observations),'observation_status_counts':status_counts,'excluded_observations':len(raw_observations)-len(obs),'unique_peptides':len(sequences),'observed_training_peptides':int(obs.peptide_id.nunique()),'folds':5,'final_model_prespecified':'esm_pc','paper_prequential_validated':False})
    return report


def predict(args):
    paths,hashes=_identity(args,['checkpoint','features','embeddings','embedding_metadata'])
    sequences,pc,design,columns=load_design(args.features,args.embeddings,args.embedding_metadata)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    target=checkpoint_target_metadata(checkpoint)
    design_target=target['target_semantics']=='design_target_not_measured'
    if checkpoint.get('task')!=TASK or checkpoint.get('output_dim')!=1:raise ValueError('Checkpoint task mismatch: requires single-output peptide_best_mic; six-strain models are incompatible')
    if checkpoint.get('activation')!='GELU':raise ValueError('Checkpoint activation must be GELU')
    if checkpoint.get('model_kind')!='esm_pc' or checkpoint['strain_ids']!=list(STRAINS) or checkpoint['feature_columns']!=columns or checkpoint['input_dim']!=design.shape[1]:raise ValueError('Checkpoint/feature identity mismatch')
    model=CalibrationMLP(checkpoint['input_dim'],checkpoint['hidden_dim']).to(args.device);model.load_state_dict(checkpoint['state_dict']);model.eval()
    prediction=infer(model,checkpoint,design,args)
    rows=[]
    for i,row in sequences.iterrows():
        for j,strain in enumerate(STRAINS):
            value=float(prediction[i,j]);mic=float(np.exp2(value))
            if not np.isfinite(mic):raise RuntimeError('MIC exponent overflow')
            rows.append({'task':TASK,'peptide_id':row.peptide_id,'sequence':row.sequence,('predicted_design_target_log2_mic' if design_target else 'predicted_log2_mic'):value,('predicted_design_target_mic_ug_ml' if design_target else 'predicted_mic_ug_ml'):mic,('design_target_threshold_32_score' if design_target else 'threshold_32_score'):float(expit(5-value)),('design_target_threshold_256_score' if design_target else 'threshold_256_score'):float(expit(8-value)),**target,'threshold_temperature_log2':1.,'probability_calibrated':False,'paper_prequential_validated':False})
    with fresh_output(args.output_dir) as output:
        pd.DataFrame(rows).to_csv(output/'predictions.csv',index=False)
        _finish(output,paths,hashes,{'task':TASK,'output_dim':1,**target,'mode':'predict','status':'complete','config':vars(args),'strain_ids':list(STRAINS),'paper_prequential_validated':False})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['train','predict'],default='train')
    for name in ('sequences','observations','checkpoint'):parser.add_argument('--'+name)
    for name in ('features','embeddings','embedding-metadata','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--epochs',type=int,default=200);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--learning-rate',type=float,default=5e-5);parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=11);parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--hidden-dim',type=int,default=16);parser.add_argument('--bootstrap-replicates',type=int,default=1000)
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {'train','predict'}:argv=['--mode',argv[0]]+argv[1:]
    args=parser.parse_args(argv)
    if any(getattr(args,name)<1 for name in ('epochs','batch_size','threads','hidden_dim','bootstrap_replicates')) or args.learning_rate<=0 or args.weight_decay<0:parser.error('Invalid optimization/count parameter')
    required=('sequences','observations') if args.mode=='train' else ('checkpoint',)
    if any(not getattr(args,name) for name in required):parser.error('Mode requires '+', '.join('--'+name for name in required))
    torch.set_num_threads(args.threads)
    (train if args.mode=='train' else predict)(args)


if __name__=='__main__':main()
