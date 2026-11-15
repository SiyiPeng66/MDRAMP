"""Cache frozen ESM3 designs for explicit preference-head fitting."""
import argparse,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch,yaml
from mdramp.prospective_calibration import design
from mdramp.calibration_train import CalibrationMLP
from src.features.physicochemical import compute_descriptors
from src.features.scaler import StateFeatureScaler
from src.features.esm3_encoder import ESM3Encoder

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads)
 root=Path(__file__).resolve().parents[1];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 cfg=root/'configs/prospective575/calibration.yaml';opts=yaml.safe_load(cfg.read_text())['options'];args={k.replace('-','_'):str((cfg.parent/v).resolve()) if k in ['features','embeddings','embedding-metadata','expert-means','expert-manifest','scaler'] else v for k,v in opts.items()}
 seq,x,_,em=design(SimpleNamespace(**args));lookup=dict(zip(seq.sequence,x))
 source=root/'outputs/prospective_575_round100_v1/score_rank_audit/all_rounds_scores_ranked.csv';frame=pd.read_csv(source)
 origpath=root/'outputs/prospective_575_round100_v1/score_rank_audit/all230_scored.csv';orig=pd.read_csv(origpath)
 extras=orig[~orig.sequence.isin(frame.sequence)].copy();frame=pd.concat([frame,extras],ignore_index=True)
 assert len(frame)==660 and frame.sequence.nunique()==660
 sc=json.loads((root/'configs/prospective575/scorer.json').read_text());scaler=StateFeatureScaler.load(sc['scaler']);cache=out/'esm_cache';cache.mkdir(exist_ok=True)
 binding={'esm_checkpoint_sha256':sha(sc['esm_checkpoint']), 'encoder_implementation_sha256':sha(root/'src/features/esm3_encoder.py')}
 bindpath=cache/'binding.json'
 if bindpath.exists() and json.loads(bindpath.read_text())!=binding:raise ValueError('Cached encoder binding differs')
 if not bindpath.exists() and list(cache.glob('*.npy')):raise ValueError('Unbound existing cache requires explicit audit')
 bindpath.write_text(json.dumps(binding,indent=2));encoder=None;vectors=[]
 for i,r in frame.iterrows():
  s=r.sequence
  if s in lookup:vectors.append(lookup[s]);continue
  path=cache/(hashlib.sha256(s.encode()).hexdigest()+'.npy')
  if path.exists():v=np.load(path,allow_pickle=False)
  else:
   if encoder is None:encoder=ESM3Encoder(device='cpu',weights_path=sc['esm_checkpoint'])
   v=encoder.encode_sequence(s,s).global_embedding;np.save(path,v)
  d=pd.DataFrame([compute_descriptors(s)]);assert list(d)==scaler.columns
  vectors.append(np.concatenate([v,scaler.transform(d.to_numpy(dtype=np.float32))[0],r[['mu_prior','mu_mem','mu_int']].to_numpy(float)]))
  if i%20==0:print(f'encoded {i+1}/660',flush=True)
 values=np.asarray(vectors,dtype=np.float32);checkpoint=torch.load(sc['mic_checkpoint'],map_location='cpu',weights_only=True);model=CalibrationMLP(1571,16);model.load_state_dict(checkpoint['state_dict']);model.eval()
 with torch.no_grad():baseline=model(torch.from_numpy(values))[:,0].numpy()
 error=float(np.max(np.abs(baseline-frame.predicted_adjusted_log2_mic.to_numpy())))
 print('baseline max log2 error',error,flush=True)
 if error>1e-4:raise ValueError('Cached design does not reproduce baseline')
 policy_path=root/'configs/prospective575/scenario_requested_ranks.json';policy=json.loads(policy_path.read_text())
 divisors=[policy['sequence_divisors'].get(r.sequence,policy['round_divisors'].get(r['round'],policy['default_divisor'])) for _,r in frame.iterrows()]
 result=frame[['sequence','round','p_cons']].copy();result['baseline_log2']=baseline;result['target_log2']=baseline+np.log2(10/np.asarray(divisors));result['is_original']=result.sequence.isin(seq.sequence);result['group_in_600']=result.sequence.isin(pd.read_csv(source).sequence);result['requested_rank']=result.sequence.map({'FLRRLVKWLKKM':1,'MLKALKNLIRKLL':5,'LRKKWLRRLLKFM':8})
 result.to_csv(out/'design.csv',index=False);np.savez_compressed(out/'design.npz',x=values,sequences=result.sequence.to_numpy(dtype=str))
 (out/'manifest.json').write_text(json.dumps({'baseline_max_log2_error':error,'frozen_artifact_sha256':em['frozen_artifact_sha256'],'input_sha256':{str(p):sha(p) for p in [source,origpath,policy_path,Path(sc['mic_checkpoint']),Path(__file__)]},'output_sha256':{p.name:sha(p) for p in [out/'design.csv',out/'design.npz']},'target_semantics':'explicit_user_preference_targets_not_measured_MIC'},indent=2))
if __name__=='__main__':main()
