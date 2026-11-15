"""Shared frozen expert evaluation for prospective calibration and generation."""
import json
from pathlib import Path
import numpy as np
import torch
from src.score.frozen_scorer import FrozenSequenceScorer
from src.score.score_candidates import load_ensemble
from src.features.scaler import StateFeatureScaler
from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import compute_descriptors
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import ProjectedPairAffinityExpert

KEYS=('esm_checkpoint','scaler','amp_prior','membrane','target_affinity','target_embeddings','target_embedding_metadata','panel_manifest')


def load_experts(config,hashes,device='cpu',*,strict_training=True):
    scaler=StateFeatureScaler.load(config['scaler'])
    if scaler.columns!=list(compute_descriptors('ACDEFGHIKLMNPQRSTVWY')) or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or not (scaler.scale>0).all():raise ValueError('Invalid frozen descriptor scaler')
    prior,pm=load_ensemble(config['amp_prior'],'PaperAMPPriorExpert')
    membrane,mm=load_ensemble(config['membrane'],'FeaturewiseGatedMembraneExpert')
    saved=torch.load(config['target_affinity'],map_location='cpu',weights_only=True);im=saved['metadata']
    for key,meta in [('amp_prior',pm),('membrane',mm),('target_affinity',im)]:
        if strict_training:
            FrozenSequenceScorer._validate_expert_training(key,meta)
        else:
            training=meta.get('training_config',{})
            if any(not np.isfinite(float(training.get(k,0))) or float(training.get(k,0))<=0 for k in ('epochs','batch_size','learning_rate')):
                raise ValueError('Missing valid public expert training provenance: '+key)
        if meta.get('seeds')!=[11,23,37,51,73]:raise ValueError('V6 five fixed seeds required')
    if pm.get('input_dim')!=1568 or (mm.get('esm_dim'),mm.get('physchem_dim'))!=(1536,32):raise ValueError('Invalid expert dimensions')
    if im.get('architecture')!='ProjectedPairAffinityExpert' or tuple(im.get(k) for k in ('peptide_dim','target_dim','physchem_dim'))!=(1536,1536,32):raise ValueError('Invalid interaction architecture')
    interaction=ExpertEnsemble(lambda:ProjectedPairAffinityExpert(1536,1536,32,projection_dim=im.get('projection_dim',256),rank=im.get('bilinear_rank',64)),seeds=im['seeds'])
    interaction.load_state_dict(saved['state_dict'])
    for model in (prior,membrane,interaction):model.to(device).eval().requires_grad_(False)
    store=EmbeddingStore.load(config['target_embeddings'],config['target_embedding_metadata'])
    try:
        frame=store.metadata;ids=frame.record_id.astype(str).tolist();idx=frame.global_index.to_numpy(float)
        if len(ids)<5 or len(set(ids))!=len(ids) or not np.isfinite(idx).all() or not np.equal(idx,idx.astype(int)).all() or len(set(idx))!=len(idx) or (idx<0).any() or (idx>=len(store.global_embeddings)).any():raise ValueError('Invalid panel index')
        panel=json.loads(Path(config['panel_manifest']).read_text())
        if panel.get('status')!='frozen_new_prospective_panel' or panel.get('historical_r0_panel') is not False:raise ValueError('Explicit new prospective panel provenance required')
        if panel.get('record_ids')!=ids:raise ValueError('Panel target order mismatch')
        for key in ('target_embeddings','target_embedding_metadata','esm_checkpoint'):
            if panel.get('artifact_sha256',{}).get(key)!=hashes[key]:raise ValueError('Panel hash binding mismatch')
        targets=torch.as_tensor(store.global_by_record_id(ids),device=device)
        if targets.shape!=(len(ids),1536) or not torch.isfinite(targets).all() or len(torch.unique(targets,dim=0))!=len(ids):raise ValueError('Invalid or duplicate panel embedding')
    finally:store.npz.close()
    return scaler,prior,membrane,interaction,targets,ids


@torch.inference_mode()
def score_experts(pep,pc,prior,membrane,interaction,targets,target_ids,pair_batch_size=1024):
    if pair_batch_size<1:raise ValueError('Positive pair batch size required')
    pr=prior(torch.cat([pep,pc],1));me=membrane(pep,pc)
    values=[];best_ids=[];best_scores=[]
    for i in range(len(pep)):
        member_values=[];pairs=[]
        for member in interaction.members:
            logits=[]
            for start in range(0,len(targets),pair_batch_size):
                target=targets[start:start+pair_batch_size];n=len(target)
                logits.append(member.pair_logits(pep[i:i+1].expand(n,-1),target,pc[i:i+1].expand(n,-1)))
            logits=torch.cat(logits);pairs.append(logits.sigmoid());top=logits.topk(5).values;q=top.sigmoid()
            member_values.append(.8*(top.softmax(0)*q).sum()+.2*(1-(1-q).prod()))
        mean=torch.stack(pairs).mean(0);index=int(mean.argmax());best_ids.append(target_ids[index]);best_scores.append(mean[index]);values.append(torch.stack(member_values))
    values=torch.stack(values,1)
    return dict(mu_prior=pr['mean'],sigma_prior=pr['std'],mu_mem=me['mean'],sigma_mem=me['std'],mu_int=values.mean(0),sigma_int=values.std(0,unbiased=True),top_target_score=torch.stack(best_scores),top_target_id=best_ids)
