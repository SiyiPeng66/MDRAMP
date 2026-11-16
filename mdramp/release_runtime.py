"""Frozen sequence features shared by public foundation and calibrated MIC heads."""
import numpy as np
import pandas as pd
import torch
from mdramp.prospective_experts import load_experts,score_experts,KEYS
from src.features.esm3_encoder import ESM3Encoder
from src.features.physicochemical import compute_descriptors
from src.models.gated_integrator import integrate_numpy
from src.utils.hash import sha256_file
class Runtime:
 def __init__(self,bundle):
  self.config=bundle['config'];self.device=torch.device(self.config.get('device','cpu'));self.hashes={k:sha256_file(self.config[k]) for k in (*KEYS,'mic_checkpoint') if k in self.config};self.checkpoint=None;self.model=None
  if bundle['checkpoint']:
   from mdramp.preference_scorer import PreferenceSequenceScorer
   checked=PreferenceSequenceScorer(bundle['scorer_config']);self.checkpoint=checked.checkpoint
   from mdramp.preference_head import model_from_checkpoint
   self.model=model_from_checkpoint(self.checkpoint).to(self.device)
  self.scaler,self.prior,self.membrane,self.interaction,self.targets,self.target_ids=load_experts(self.config,self.hashes,self.device,strict_training=False)
  self.encoder=ESM3Encoder(device=str(self.device),weights_path=self.config['esm_checkpoint'])
 def extract(self,sequences):
  vectors=np.stack([self.encoder.encode_sequence(str(i),s).global_embedding for i,s in enumerate(sequences)])
  d=pd.DataFrame([compute_descriptors(s) for s in sequences]);pc=self.scaler.transform(d[self.scaler.columns].to_numpy(dtype=np.float32))
  with torch.inference_mode():e=score_experts(torch.from_numpy(vectors).to(self.device),torch.from_numpy(pc).to(self.device),self.prior,self.membrane,self.interaction,self.targets,self.target_ids,pair_batch_size=128)
  values={k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in e.items()};g=integrate_numpy(values['mu_prior'],values['mu_mem'],values['mu_int'],prior_std=values['sigma_prior'],mem_std=values['sigma_mem'],aff_std=values['sigma_int'])
  frame=pd.DataFrame({'sequence':sequences,**values,**g});x=np.concatenate([vectors,pc,frame[['mu_prior','mu_mem','mu_int']].to_numpy(dtype=np.float32)],axis=1).astype(np.float32)
  return x,frame
 def predict(self,frame):
  x,e=self.extract(list(dict.fromkeys(frame.sequence)));result=e.copy()
  if self.model is not None:
   with torch.inference_mode():y=self.model(torch.from_numpy(x).to(next(self.model.parameters()).device)).squeeze(-1).cpu().numpy()
   result['predicted_mic']=np.exp2(y.astype(float));result['R_t']=result.p_cons/(1+np.exp(y.astype(float)-5))
  return frame.merge(result,on='sequence',validate='many_to_one',sort=False)
 def verify(self):
  if any(sha256_file(self.config[k])!=v for k,v in self.hashes.items()):raise ValueError('Model assets changed during execution')
def load_runtime(bundle):return Runtime(bundle)
