"""Compute actual frozen expert means without any MIC checkpoint."""
import argparse
import json
import os
import fcntl
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.features.embedding_store import EmbeddingStore
from src.utils.hash import sha256_file
from src.models.gated_integrator import GateParams, integrate_numpy
from mdramp.prospective_experts import KEYS,load_experts,score_experts


def validate_embedding_provenance(provenance_path, embeddings, metadata, esm_hash):
    provenance=json.loads(Path(provenance_path).read_text())
    if provenance.get('status')!='complete' or provenance.get('encoder')!='ESM3' or provenance.get('embedding_dimension')!=1536:
        raise ValueError('Complete actual ESM3 embedding provenance required')
    if provenance.get('binding',{}).get('checkpoint_sha256')!=esm_hash:
        raise ValueError('Peptide embedding ESM checkpoint mismatch')
    for path in (embeddings,metadata):
        if provenance.get('output_sha256',{}).get(Path(path).name)!=sha256_file(path):
            raise ValueError('Peptide embedding artifact provenance mismatch')
    return provenance


def aligned_embeddings(store,frame,id_column):
    meta=store.metadata
    if not {'record_id','sequence','global_index'}.issubset(meta) or meta.record_id.isna().any() or meta.record_id.astype(str).duplicated().any():
        raise ValueError('Unique embedding IDs and sequence provenance required')
    index=pd.to_numeric(meta.global_index,errors='raise').to_numpy(float)
    if not np.isfinite(index).all() or not np.equal(index,index.astype(int)).all() or len(set(index))!=len(index) or (index<0).any() or (index>=len(store.global_embeddings)).any():
        raise ValueError('Invalid embedding index')
    if store.global_embeddings.shape!=(len(meta),1536) or not np.isfinite(store.global_embeddings).all():
        raise ValueError('Invalid real peptide embedding array')
    mapping=meta.assign(record_id=meta.record_id.astype(str)).set_index('record_id')
    ids=frame[id_column].astype(str).tolist()
    if set(ids)!=set(mapping.index):raise ValueError('Peptide embedding and feature ID sets differ')
    aligned=mapping.loc[ids]
    if aligned.sequence.isna().any() or aligned.sequence.astype(str).tolist()!=frame.sequence.astype(str).tolist():
        raise ValueError('Peptide sequence identity differs between features and embeddings')
    return store.global_by_record_id(ids)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scorer-config','features','embeddings','embedding-metadata','embedding-provenance','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--id-column',default='peptide_id');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--pair-batch-size',type=int,default=1024);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if min(a.batch_size,a.pair_batch_size)<1:raise ValueError('Batch sizes must be positive')
    path=Path(a.scorer_config).absolute();config=json.loads(path.read_text())
    config={k:str(Path(config[k]) if Path(config[k]).is_absolute() else path.parent/config[k]) for k in KEYS}
    frozen={k:sha256_file(v) for k,v in config.items()}
    inputs={str(Path(v).absolute()):sha256_file(v) for v in (a.scorer_config,a.features,a.embeddings,a.embedding_metadata,a.embedding_provenance,__file__,str(Path(__file__).with_name('prospective_experts.py')))}
    out=Path(a.output_dir).absolute()
    if out.exists():raise ValueError('Output exists')
    scaler,prior,membrane,interaction,targets,ids=load_experts(config,frozen,a.device)
    frame=pd.read_csv(a.features)
    if not {'sequence',a.id_column,*scaler.columns}.issubset(frame):raise ValueError('Missing sequence, identifiers or descriptors')
    if frame.empty or frame[a.id_column].isna().any() or frame[a.id_column].duplicated().any() or frame.sequence.duplicated().any():raise ValueError('Unique nonempty input required')
    provenance=validate_embedding_provenance(a.embedding_provenance,a.embeddings,a.embedding_metadata,frozen['esm_checkpoint'])
    store=EmbeddingStore.load(a.embeddings,a.embedding_metadata)
    try:
        if provenance.get('record_count')!=len(store.metadata):raise ValueError('ESM record count mismatch')
        pep=aligned_embeddings(store,frame,a.id_column);pc=scaler.transform(frame[scaler.columns].to_numpy(np.float32))
        if pep.shape!=(len(frame),1536) or not np.isfinite(pep).all() or not np.isfinite(pc).all():raise ValueError('Invalid actual features')
    finally:store.npz.close()
    rows=[]
    for start in range(0,len(frame),a.batch_size):
        result=score_experts(torch.tensor(pep[start:start+a.batch_size],device=a.device),torch.tensor(pc[start:start+a.batch_size],device=a.device),prior,membrane,interaction,targets,ids,a.pair_batch_size)
        result={k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in result.items()};batch=frame.iloc[start:start+a.batch_size]
        result.update(peptide_id=batch[a.id_column].astype(str).tolist(),sequence=batch.sequence.tolist())
        result.update(integrate_numpy(result['mu_prior'],result['mu_mem'],result['mu_int'],GateParams(.5,10.,.3),result['sigma_prior'],result['sigma_mem'],result['sigma_int'],1.))
        rows.extend(pd.DataFrame(result).to_dict('records'));print(f'expert means {min(start+a.batch_size,len(frame))}/{len(frame)}',flush=True)
    output=pd.DataFrame(rows)
    if not np.isfinite(output.select_dtypes(include='number').to_numpy()).all():raise ValueError('Nonfinite expert output')
    out.parent.mkdir(parents=True,exist_ok=True)
    with (out.parent/(out.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():raise ValueError('Output exists')
        staging=Path(tempfile.mkdtemp(prefix='.'+out.name,dir=out.parent))
        try:
            output.to_csv(staging/'expert_means.csv',index=False)
            if any(sha256_file(v)!=frozen[k] for k,v in config.items()) or any(sha256_file(v)!=h for v,h in inputs.items()):raise ValueError('Input changed during expert scoring')
            manifest={'schema_version':1,'purpose':'new_prospective_frozen_expert_features','historical_reproduction':False,'record_count':len(output),'output_sha256':sha256_file(staging/'expert_means.csv'),'frozen_artifact_sha256':frozen,'input_sha256':inputs,'expert_means_order':['mu_prior','mu_mem','mu_int'],'no_mic_model_used':True}
            (staging/'manifest.json').write_text(json.dumps(manifest,indent=2));os.rename(staging,out)
        finally:
            if staging.exists():shutil.rmtree(staging)


if __name__=='__main__':main()
