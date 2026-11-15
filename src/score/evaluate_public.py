"""Held-out public expert evaluation, without a MIC head or deployment scores.

AMP-only test records receive evidence summaries, not classification accuracy.
Pair metrics distinguish source positives from operational background only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import feature_columns
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import ProjectedPairAffinityExpert
from src.score.score_candidates import load_ensemble
from src.utils.hash import sha256_file

INPUT_NAMES=('public_features','public_embeddings','public_embedding_metadata','amp_prior','membrane','scaler',
             'pair_scores','pair_features','pair_embeddings','pair_embedding_metadata','target_embeddings',
             'target_embedding_metadata','interaction_checkpoint')


def _frame(path,required,name):
    frame=pd.read_csv(path,dtype={'peptide_id':str,'target_id':str,'record_id':str,'sequence':str,'target_sequence':str,'split':str})
    if not set(required).issubset(frame):raise ValueError(f'{name} requires columns {sorted(set(required)-set(frame))}')
    if frame[list(required)].isna().any().any():raise ValueError(f'{name} contains missing required values')
    return frame


def _store(embeddings,metadata):
    store=EmbeddingStore.load(embeddings,metadata)
    required={'record_id','global_index','sequence'}
    if not required.issubset(store.metadata):raise ValueError('Embedding metadata requires record_id/global_index/sequence')
    store.metadata['record_id']=store.metadata.record_id.astype(str)
    if store.metadata.record_id.duplicated().any() or store.metadata.global_index.duplicated().any():raise ValueError('Embedding metadata IDs/indices must be unique')
    indices=pd.to_numeric(store.metadata.global_index,errors='raise').to_numpy()
    if not np.isfinite(indices).all() or not np.equal(indices,indices.astype(int)).all() or (indices<0).any() or (indices>=len(store.global_embeddings)).any():raise ValueError('Invalid embedding metadata indices')
    if store.global_embeddings.ndim!=2 or not np.isfinite(store.global_embeddings).all():raise ValueError('Invalid embedding matrix')
    return store


def _aligned(store,ids,sequences):
    lookup=store.metadata.set_index('record_id')
    if not set(ids).issubset(lookup.index):raise ValueError('Embedding cache does not cover required IDs')
    if lookup.loc[ids,'sequence'].astype(str).tolist()!=list(map(str,sequences)):raise ValueError('Sequence does not match embedding metadata for its ID')
    return store.global_by_record_id(list(ids))


def _features(frame,scaler):
    if feature_columns(frame)!=scaler.columns:raise ValueError('Descriptor columns/order do not match frozen scaler')
    values=frame[scaler.columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite descriptors')
    return scaler.transform(values)


def evaluate(args):
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    destination=Path(args.output_dir).absolute()
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    inputs={name:str(Path(getattr(args,name)).absolute()) for name in INPUT_NAMES}
    hashes={name:sha256_file(path) for name,path in inputs.items()}
    public=_frame(args.public_features,{'peptide_id','sequence','split'},'public features')
    if public.peptide_id.duplicated().any():raise ValueError('Public feature IDs must be unique')
    if not set(public.split).issubset({'train','validation','test'}):raise ValueError('Invalid public split values')
    public=public[public.split.eq('test')].copy().reset_index(drop=True)
    if public.empty:raise ValueError('Public features contain no test records')
    pairs=_frame(args.pair_scores,{'peptide_id','target_id','sequence','target_sequence','split','label','label_type'},'pair scores')
    if pairs.empty or not pairs.split.eq('test').all():raise ValueError('Pair evaluation accepts only explicit test rows, never train/validation/mixed input')
    labels=pd.to_numeric(pairs.label,errors='raise')
    if not labels.isin([0,1]).all() or labels.nunique()!=2:raise ValueError('Pair ROC-AUC/AP require both binary source labels 0 and 1')
    if not ((labels.eq(1)&pairs.label_type.eq('positive'))|(labels.eq(0)&pairs.label_type.eq('background'))).all():raise ValueError('Pair label and positive/background semantics disagree')
    if pairs[['peptide_id','target_id']].duplicated().any() or pairs[['sequence','target_sequence']].duplicated().any():raise ValueError('Duplicate evaluated pairs')
    pair_features=_frame(args.pair_features,{'peptide_id','sequence','split'},'pair features')
    if pair_features.peptide_id.duplicated().any():raise ValueError('Pair feature IDs must be unique')
    lookup=pair_features.set_index('peptide_id')
    if not set(pairs.peptide_id).issubset(lookup.index):raise ValueError('Missing pair feature rows')
    selected=lookup.loc[pairs.peptide_id].reset_index()
    if not selected.split.eq('test').all():raise ValueError('Pair feature split contradicts held-out test declaration')
    if selected.sequence.tolist()!=pairs.sequence.tolist():raise ValueError('Pair sequences disagree with feature records')
    if 'cluster_id' in pairs and 'cluster_id' in selected and selected.cluster_id.tolist()!=pairs.cluster_id.tolist():raise ValueError('Pair clusters disagree with feature records')
    scaler=StateFeatureScaler.load(args.scaler)
    public_store=_store(args.public_embeddings,args.public_embedding_metadata)
    pair_store=_store(args.pair_embeddings,args.pair_embedding_metadata)
    target_store=_store(args.target_embeddings,args.target_embedding_metadata)
    temporary=None
    destination.parent.mkdir(parents=True,exist_ok=True)
    lock=destination.parent/('.'+destination.name+'.evaluation.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    try:
        if destination.exists():raise FileExistsError(destination)
        temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
        amp,amp_meta=load_ensemble(args.amp_prior,'PaperAMPPriorExpert')
        membrane,mem_meta=load_ensemble(args.membrane,'FeaturewiseGatedMembraneExpert')
        amp.to(args.device).eval();membrane.to(args.device).eval()
        public_vectors=_aligned(public_store,public.peptide_id.tolist(),public.sequence.tolist())
        public_pc=_features(public,scaler);public_result=public[['peptide_id','sequence','split']].copy()
        prior_means=[];prior_stds=[];mem_means=[];mem_stds=[]
        with torch.inference_mode():
            for start in range(0,len(public),args.batch_size):
                end=start+args.batch_size
                vectors=torch.as_tensor(public_vectors[start:end],device=args.device)
                descriptors=torch.as_tensor(public_pc[start:end],device=args.device)
                prior=amp(torch.cat([vectors,descriptors],dim=1));mem=membrane(vectors,descriptors)
                prior_means.extend(prior['mean'].cpu().tolist());prior_stds.extend(prior['std'].cpu().tolist())
                mem_means.extend(mem['mean'].cpu().tolist());mem_stds.extend(mem['std'].cpu().tolist())
        public_result['mu_prior']=prior_means;public_result['sigma_prior']=prior_stds
        public_result['mu_mem']=mem_means;public_result['sigma_mem']=mem_stds
        if not np.isfinite(public_result[['mu_prior','sigma_prior','mu_mem','sigma_mem']].to_numpy()).all():raise ValueError('Nonfinite public expert predictions')
        public_result.to_csv(temporary/'public_test_expert_scores.csv',index=False)
        del amp,membrane
        checkpoint=torch.load(args.interaction_checkpoint,map_location='cpu',weights_only=False)
        meta=checkpoint['metadata']
        if meta.get('architecture')!='ProjectedPairAffinityExpert' or len(set(meta.get('seeds',[])))!=5:raise ValueError('Expected five-member interaction checkpoint')
        interaction=ExpertEnsemble(lambda:ProjectedPairAffinityExpert(meta['peptide_dim'],meta['target_dim'],meta['physchem_dim'],projection_dim=meta.get('projection_dim',256),rank=meta.get('bilinear_rank',64)),seeds=meta['seeds'])
        interaction.load_state_dict(checkpoint['state_dict']);interaction.to(args.device).eval()
        pv=_aligned(pair_store,pairs.peptide_id.tolist(),pairs.sequence.tolist())
        tv=_aligned(target_store,pairs.target_id.tolist(),pairs.target_sequence.tolist())
        pc=_features(selected,scaler);means=[];stds=[]
        with torch.inference_mode():
            for start in range(0,len(pairs),args.batch_size):
                end=start+args.batch_size
                result=interaction(torch.as_tensor(pv[start:end],device=args.device),torch.as_tensor(tv[start:end],device=args.device),torch.as_tensor(pc[start:end],device=args.device))
                means.extend(result['mean'].cpu().tolist());stds.extend(result['std'].cpu().tolist())
        if not np.isfinite(means).all() or not np.isfinite(stds).all():raise ValueError('Nonfinite pair predictions')
        pair_result=pairs.copy();pair_result['mu_interaction']=means;pair_result['sigma_interaction']=stds
        pair_result['evaluation_label_semantics']=np.where(labels.eq(1),'source_positive','operational_background_not_confirmed_nonbinding')
        pair_result.to_csv(temporary/'pair_test_scores.csv',index=False)
        metrics={'schema_version':1,
                 'interaction_curation': {key: meta.get(key) for key in ('data_mode','v6_pair_curation_complete','curation_manifest_sha256')},
                 'public_test':{'records':len(public),'classification_metrics':None,
                 'reason':'AMP-only public test set; no matched negative benchmark, so no accuracy/AUC/AP is reported.'},
                 'pair_test':{'records':len(pairs),'positive_count':int(labels.sum()),'background_count':int(labels.eq(0).sum()),
                              'roc_auc':float(roc_auc_score(labels,means)),'average_precision':float(average_precision_score(labels,means)),
                              'label_semantics':'source positives versus operational background; background is not confirmed nonbinding'},
                 'limitations':['No MIC head, predicted MIC, A_t or R_t is computed.','No threshold fitting, calibration, checkpoint selection or training update uses these test outcomes.',
                                'Pair metrics measure cold-peptide discrimination only; they do not establish clinical MDR activity or causal protein mechanisms.',
                                'This new downstream split is not established as independent of the upstream TPepPro teacher training set; metrics are internal public-data diagnostics.',
                                'Split labels and ID/sequence alignment are validated; historical training provenance must additionally be audited from training artifacts.']}
        (temporary/'metrics.json').write_text(json.dumps(metrics,indent=2))
        after={name:sha256_file(path) for name,path in inputs.items()}
        if after!=hashes:raise RuntimeError('Evaluation inputs or weights changed during prediction')
        manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'scope':'public_test_only_no_feedback',
                  'inputs':inputs,'input_sha256':hashes,'configuration':{'batch_size':args.batch_size,'device':args.device},
                  'output_sha256':{p.name:sha256_file(p) for p in sorted(temporary.iterdir())},'metrics':metrics}
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if destination.exists():raise FileExistsError(destination)
        os.rename(temporary,destination);temporary=None
        return manifest
    finally:
        for store in (public_store,pair_store,target_store):store.npz.close()
        if temporary is not None:shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in INPUT_NAMES:parser.add_argument('--'+name.replace('_','-'),required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();print(json.dumps(evaluate(args),indent=2))
if __name__=='__main__':main()
