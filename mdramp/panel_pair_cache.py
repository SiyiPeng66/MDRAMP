"""Resumable real pair-logit cache for score-independent panel stability probes."""
import argparse,fcntl,json,os,tempfile,platform,importlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.utils.hash import sha256_file
from src.features.embedding_store import EmbeddingStore
from src.features.physicochemical import compute_descriptors
from src.features.scaler import StateFeatureScaler
from src.score.score_candidates import load_ensemble
from src.score.frozen_scorer import FrozenSequenceScorer
from src.models.expert_modules import ProjectedPairAffinityExpert
from src.models.ensemble import ExpertEnsemble
from mdramp.prospective_features import validate_embedding_provenance,aligned_embeddings


@torch.inference_mode()
def projected_logits(member,peptide_projected,target_projected,pc):
    peptide=peptide_projected.expand(len(target_projected),-1)
    features=torch.cat([abs(peptide-target_projected),peptide*target_projected,pc.expand(len(target_projected),-1),member.left_rank(peptide)*member.right_rank(target_projected)],dim=1)
    return member.head(features).squeeze(-1)


def atomic_npz(path,**arrays):
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name);os.close(fd)
    try:
        with open(name,'wb') as handle:np.savez(handle,**arrays)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def validate_probe_cache(path,record_id,target_count):
    side=json.loads(path.with_suffix('.json').read_text())
    if side.get('record_id')!=str(record_id):raise ValueError('Probe cache identity mismatch')
    if side.get('sha256')!=sha256_file(path):raise ValueError('Probe cache changed')
    with np.load(path,allow_pickle=False) as item:
        if set(item.files)!={'logits','fixed'}:raise ValueError('Invalid probe cache keys')
        logits=item['logits'];fixed=item['fixed']
        if logits.shape!=(5,target_count) or fixed.shape!=(4,):raise ValueError('Invalid probe cache shape')
        if not np.isfinite(logits).all() or not np.isfinite(fixed).all():raise ValueError('Nonfinite probe cache')
        return logits.copy(),fixed.copy()


def dependency_paths():
    modules=('mdramp.prospective_features','src.models.expert_modules','src.score.frozen_scorer',
             'src.features.physicochemical','src.features.scaler','src.models.ensemble',
             'src.score.score_candidates','src.features.embedding_store','src.utils.hash','src.train.train_utils')
    return [Path(__file__).resolve()]+[Path(importlib.import_module(name).__file__).resolve() for name in modules]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scorer-config','proteins','protein-embeddings','protein-metadata','protein-provenance','probes','probe-embeddings','probe-metadata','probe-provenance','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--pair-batch-size',type=int,default=512);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.pair_batch_size<1:raise ValueError('Positive batch size required')
    config_path=Path(a.scorer_config).absolute();config=json.loads(config_path.read_text());keys=('esm_checkpoint','scaler','amp_prior','membrane','target_affinity')
    config={k:str(Path(config[k]) if Path(config[k]).is_absolute() else config_path.parent/config[k]) for k in keys}
    inputs={str(Path(v).absolute()):sha256_file(v) for k,v in vars(a).items() if k not in ('resume','output_dir','pair_batch_size','device')}
    inputs.update({str(Path(v).absolute()):sha256_file(v) for v in config.values()})
    for file in dependency_paths():inputs[str(file)]=sha256_file(file)
    binding={'input_sha256':inputs,'device':a.device,'pair_batch_size':a.pair_batch_size,'torch':str(torch.__version__),'python':platform.python_version(),'layout':'probe,member,target','seeds':[11,23,37,51,73]}
    out=Path(a.output_dir).absolute()
    if out.exists() and not a.resume:raise ValueError('Existing output requires explicit --resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bp=out/'binding.json'
        if bp.exists():
            if json.loads(bp.read_text())!=binding:raise ValueError('Resume binding changed')
        else:
            if a.resume and any(out.iterdir()):
                if set(x.name for x in out.iterdir())!={'.lock'}:raise ValueError('Unbound existing cache')
            bp.write_text(json.dumps(binding,indent=2))
        if (out/'manifest.json').exists():
            manifest=json.loads((out/'manifest.json').read_text())
            if any(sha256_file(out/k)!=v for k,v in manifest['output_sha256'].items()):raise ValueError('Completed cache changed')
            print('Verified complete cache');return
        frames=[];arrays=[]
        for prefix in ('protein','probe'):
            csv=a.proteins if prefix=='protein' else a.probes;frame=pd.read_csv(csv)
            if not {'record_id','sequence'}.issubset(frame) or frame.empty or frame.record_id.isna().any() or frame.record_id.duplicated().any():raise ValueError('Unique nonempty probe/protein IDs required')
            if not frame.sequence.astype(str).str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+').all():raise ValueError('Invalid biological sequences')
            npz=getattr(a,prefix+'_embeddings');metadata=getattr(a,prefix+'_metadata')
            provenance=validate_embedding_provenance(getattr(a,prefix+'_provenance'),npz,metadata,sha256_file(config['esm_checkpoint']))
            store=EmbeddingStore.load(npz,metadata)
            try:
                if provenance.get('record_count')!=len(frame):raise ValueError('Embedding provenance record count mismatch')
                arrays.append(aligned_embeddings(store,frame,'record_id'))
            finally:store.npz.close()
            frames.append(frame)
        proteins,probes=frames;target,pep=[torch.tensor(x,device=a.device) for x in arrays]
        scaler=StateFeatureScaler.load(config['scaler']);descriptors=pd.DataFrame([compute_descriptors(s) for s in probes.sequence])
        if scaler.columns!=list(descriptors):raise ValueError('Descriptor column mismatch')
        pc=torch.tensor(scaler.transform(descriptors.to_numpy(np.float32)),device=a.device)
        if not torch.isfinite(pc).all():raise ValueError('Invalid scaled descriptors')
        prior,pm=load_ensemble(config['amp_prior'],'PaperAMPPriorExpert');membrane,mm=load_ensemble(config['membrane'],'FeaturewiseGatedMembraneExpert')
        saved=torch.load(config['target_affinity'],map_location='cpu',weights_only=True);im=saved['metadata']
        for key,meta in [('amp_prior',pm),('membrane',mm),('target_affinity',im)]:
            FrozenSequenceScorer._validate_expert_training(key,meta)
            if meta.get('seeds')!=binding['seeds']:raise ValueError('Wrong ensemble seeds')
        if pm.get('input_dim')!=1568 or (mm.get('esm_dim'),mm.get('physchem_dim'))!=(1536,32) or im.get('architecture')!='ProjectedPairAffinityExpert' or tuple(im.get(k) for k in ('peptide_dim','target_dim','physchem_dim'))!=(1536,1536,32):raise ValueError('Wrong expert architecture')
        interaction=ExpertEnsemble(lambda:ProjectedPairAffinityExpert(1536,1536,32,im.get('projection_dim',256),im.get('bilinear_rank',64)),seeds=binding['seeds']);interaction.load_state_dict(saved['state_dict'])
        for model in (prior,membrane,interaction):model.to(a.device).eval().requires_grad_(False)
        cache=out/'probes_cache';cache.mkdir(exist_ok=True);errors=[]
        with torch.inference_mode():
            projections=[(m.peptide_norm(m.peptide_projection(pep)),m.target_norm(m.target_projection(target))) for m in interaction.members]
            for m,(pp,tp) in zip(interaction.members,projections):
                for i in sorted(set([0,len(probes)//2,len(probes)-1])):
                    chosen=torch.tensor(sorted(set([0,len(proteins)//2,len(proteins)-1])),device=a.device)
                    direct=m.pair_logits(pep[i:i+1].expand(len(chosen),-1),target[chosen],pc[i:i+1].expand(len(chosen),-1));fast=projected_logits(m,pp[i:i+1],tp[chosen],pc[i:i+1])
                    error=float((direct-fast).abs().max());errors.append(error)
                    if error>1e-5 or not torch.allclose(direct,fast,rtol=1e-5,atol=1e-5):raise ValueError('Projected pair acceleration differs from actual pair_logits')
            for i in range(len(probes)):
                path=cache/f'{i:06d}.npz';side=path.with_suffix('.json')
                if path.exists() and side.exists():
                    validate_probe_cache(path,probes.iloc[i].record_id,len(proteins))
                    continue
                logits=[]
                for m,(pp,tp) in zip(interaction.members,projections):
                    logits.append(torch.cat([projected_logits(m,pp[i:i+1],tp[j:j+a.pair_batch_size],pc[i:i+1]) for j in range(0,len(proteins),a.pair_batch_size)]).cpu().numpy())
                pr=prior(torch.cat([pep[i:i+1],pc[i:i+1]],1));me=membrane(pep[i:i+1],pc[i:i+1]);fixed=np.array([pr['mean'].item(),pr['std'].item(),me['mean'].item(),me['std'].item()])
                logits=np.stack(logits)
                if not np.isfinite(logits).all() or not np.isfinite(fixed).all():raise ValueError('Nonfinite model output')
                atomic_npz(path,logits=logits,fixed=fixed);side.write_text(json.dumps({'sha256':sha256_file(path),'record_id':str(probes.iloc[i].record_id)}))
                print(f'pair cache {i+1}/{len(probes)}',flush=True)
        temporary=out/'logits.tmp.npy';matrix=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(probes),5,len(proteins)));fixed=[]
        for i in range(len(probes)):
            path=cache/f'{i:06d}.npz'
            logits,values=validate_probe_cache(path,probes.iloc[i].record_id,len(proteins))
            matrix[i]=logits;fixed.append(values)
        matrix.flush();del matrix;os.replace(temporary,out/'logits.npy')
        probes.to_csv(out/'peptides.csv',index=False);proteins.to_csv(out/'targets.csv',index=False)
        fixed=pd.DataFrame(fixed,columns=['mu_prior','sigma_prior','mu_mem','sigma_mem']);fixed.insert(0,'record_id',probes.record_id.tolist());fixed.to_csv(out/'fixed_experts.csv',index=False)
        if any(sha256_file(k)!=v for k,v in inputs.items()) or json.loads(bp.read_text())!=binding:raise ValueError('Source or binding changed during scoring')
        manifest={'status':'complete','layout':binding['layout'],'shape':[len(probes),5,len(proteins)],'binding':binding,'acceleration_validation':{'max_absolute_error':max(errors),'tested_pairs':len(errors)*3,'tolerance':1e-5},'output_sha256':{name:sha256_file(out/name) for name in ('logits.npy','peptides.csv','targets.csv','fixed_experts.csv','binding.json')}}
        tmp=out/'manifest.tmp.json';tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,out/'manifest.json')
if __name__=='__main__':main()
