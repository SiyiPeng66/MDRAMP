"""New prospective 1571-input calibration with explicitly user-adjusted MIC intervals.

This is not historical paper reproduction. Original labels are retained and
interval-Huber supervision extends the manuscript's exact/right-censored loss.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mdramp import calibration_train as core

SEMANTICS='user_adjusted_approximate_mic'
EXPERT_COLUMNS=['mu_prior','mu_mem','mu_int']


def design(args,sequences=None):
    sequences,pc,esm_pc,columns=core.load_design(args.features,args.embeddings,args.embedding_metadata,sequences)
    if esm_pc.shape[1]!=1568:raise ValueError('Actual 1536-dimensional ESM3 plus 32 descriptors required')
    means=core._read(args.expert_means,['peptide_id','sequence',*EXPERT_COLUMNS])
    if means.peptide_id.duplicated().any():raise ValueError('Duplicate expert IDs')
    means=means.set_index('peptide_id').loc[sequences.peptide_id]
    if means.sequence.tolist()!=sequences.sequence.tolist():raise ValueError('Expert sequence mismatch')
    values=means[EXPERT_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid expert probabilities')
    manifest=json.loads(Path(args.expert_manifest).read_text())
    recorded={str(Path(path).resolve()):digest for path,digest in manifest.get('input_sha256',{}).items()}
    for path in (args.features,args.embeddings,args.embedding_metadata):
        if recorded.get(str(Path(path).resolve()))!=core.sha(path):raise ValueError('Expert feature input provenance differs from MIC training inputs')
    if manifest.get('output_sha256')!=core.sha(args.expert_means) or not manifest.get('frozen_artifact_sha256'):raise ValueError('Expert scores must be bound to frozen artifacts by manifest')
    from src.features.scaler import StateFeatureScaler
    scaler=StateFeatureScaler.load(args.scaler)
    if scaler.columns!=columns:raise ValueError('Frozen descriptor scaler columns mismatch')
    if manifest['frozen_artifact_sha256'].get('scaler')!=core.sha(args.scaler):raise ValueError('Expert and MIC scaler differ')
    esm_pc[:,-32:]=scaler.transform(pc)
    return sequences,np.concatenate([esm_pc,values],axis=1),columns,manifest


def fit_model(x,sequences,observations,args,seed):
    torch.manual_seed(seed)
    indices={pid:i for i,pid in enumerate(sequences.peptide_id)}
    values=torch.as_tensor(x[[indices[pid] for pid in observations.peptide_id]],dtype=torch.float32,device=args.device)
    lo=torch.tensor(observations.lower_log2.to_numpy(),dtype=torch.float32,device=args.device)
    hi=torch.tensor(observations.upper_log2.to_numpy(),dtype=torch.float32,device=args.device)
    model=core.CalibrationMLP(x.shape[1],16).to(args.device)
    with torch.no_grad():
        model.layers[-1].weight.zero_();model.layers[-1].bias.fill_(float(core.fit_constant(observations)[0]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    generator=torch.Generator().manual_seed(seed)
    for _ in range(args.epochs):
        for batch in torch.randperm(len(values),generator=generator).split(args.batch_size):
            batch=batch.to(args.device);optimizer.zero_grad()
            loss=core.interval_huber(model(values[batch])[:,0],lo[batch],hi[batch])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step()
    model.eval()
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'mean':[0.]*x.shape[1],'scale':[1.]*x.shape[1],'train_peptide_ids':sorted(set(observations.peptide_id)),'epochs':args.epochs,'seed':seed}
    return model,checkpoint


def train(args):
    paths,hashes=core._identity(args,['sequences','observations','features','embeddings','embedding_metadata','expert_means','expert_manifest','scaler'])
    paths['implementation']=str(Path(__file__).resolve());hashes['implementation']=core.sha(__file__)
    paths['calibration_helper']=str(Path(core.__file__).resolve());hashes['calibration_helper']=core.sha(core.__file__)
    sequences,obs=core.load_observations(args.sequences,args.observations)
    input_target = core.target_metadata(obs)
    if input_target['target_semantics'] not in ('measured_mic','user_assigned_screen_mic') or input_target['target_divisor'] != 1:
        raise ValueError('Supply original-scale measured or explicitly user-assigned screening targets; never previously divided labels')
    sequences,x,columns,expert_manifest=design(args,sequences)
    for field in ['lower_log2','upper_log2']:
        obs['original_'+field]=obs[field];obs[field]-=np.log2(10.)
    if 'exact_log2' in obs:
        obs['original_exact_log2']=obs.exact_log2;obs['exact_log2']-=np.log2(10.)
    obs['input_target_semantics']=input_target['target_semantics']
    obs['target_semantics']=SEMANTICS;obs['target_divisor']=10.
    out=obs.copy();out['constant']=np.nan;out['esm_pc']=np.nan;out['pc']=np.nan
    specification={'task':'prospective_adjusted_best_mic','target_semantics':SEMANTICS,'target_divisor':10.,'input_target_semantics':input_target['target_semantics'],'input_target_divisor':input_target['target_divisor'],'state_id':args.state_id,'historical_reproduction':False,'input_dim':1571,'hidden_dim':args.hidden_dim,'activation':'GELU','output_dim':1,'expert_columns':EXPERT_COLUMNS,'feature_columns':columns,'frozen_artifact_sha256':expert_manifest['frozen_artifact_sha256'],'input_sha256':hashes,'supervision':'interval Huber on original log2 bounds minus log2(10); an explicit extension beyond manuscript exact/right-censor supervision','normalization':'frozen public scaler on 32 descriptors only; original ESM and expert means unchanged','biological_improvement_claim':False}
    if not args.state_id or args.state_id.upper().startswith('R'):raise ValueError('Use a named new prospective state, not an R0-R5 historical state')
    with core.fresh_output(args.output_dir) as directory:
        for fold in range(5):
            training=obs[obs.fold.ne(fold)];test=obs[obs.fold.eq(fold)]
            if set(training.cluster_id)&set(test.cluster_id):raise ValueError('Cluster leakage')
            out.loc[test.index,'constant']=core.fit_constant(training)[0]
            folder=directory/f'fold_{fold}';folder.mkdir()
            for name,values in [('pc',x[:,-35:-3]),('esm_pc',x)]:
                model,checkpoint=fit_model(values,sequences,training,args,args.seed+fold)
                checkpoint.update(specification,model_kind=name,input_dim=values.shape[1],fold=fold)
                torch.save(checkpoint,folder/(name+'.pt'))
                pred=core.infer(model,checkpoint,values,args)[:,0]
                lookup=dict(zip(sequences.peptide_id,pred))
                out.loc[test.index,name]=test.peptide_id.map(lookup)
            print(f'prospective calibration fold={fold+1}/5 complete',flush=True)
        out.to_csv(directory/'oof_predictions.csv',index=False)
        report={**specification,'models':{name:core.metrics(out,name) for name in ['constant','pc','esm_pc']},'paired_cluster_bootstrap':core.bootstrap(out,args.bootstrap_replicates,args.seed),'interpretation':'OOF compares adjusted-label prediction; division by ten is a user assumption, not observed potency improvement. Final full model is prespecified.'}
        core._json(directory/'metrics.json',report)
        model,checkpoint=fit_model(x,sequences,obs,args,args.seed)
        checkpoint.update(specification,model_kind='prospective_1571',fold='all')
        torch.save(checkpoint,directory/'model.pt')
        if any(core.sha(path)!=hashes[key] for key,path in paths.items()):raise ValueError('Input changed during training')
        core._json(directory/'manifest.json',{**specification,'config':vars(args),'output_sha256':{str(p.relative_to(directory)):core.sha(p) for p in directory.rglob('*') if p.is_file()}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sequences','observations','features','embeddings','embedding-metadata','expert-means','expert-manifest','scaler','output-dir','state-id']:p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--hidden-dim',type=int,default=16);p.add_argument('--seed',type=int,default=11);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--bootstrap-replicates',type=int,default=1000)
    args=p.parse_args()
    if args.hidden_dim!=16 or min(args.epochs,args.batch_size,args.threads,args.bootstrap_replicates)<1 or args.learning_rate<=0 or args.weight_decay<0:raise ValueError('Invalid settings; prospective head requires hidden_dim=16')
    torch.set_num_threads(args.threads);train(args)

if __name__=='__main__':main()
