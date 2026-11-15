"""Explicit NEW prospective scorer; never a relaxation of historical validation.

Config binds actual frozen expert/panel artifacts and a user-adjusted 1571 head.
The strict src.score.frozen_scorer implementation is not modified.
"""
import json
from pathlib import Path
import torch
from torch import nn
from src.score.frozen_scorer import FrozenSequenceScorer
from src.utils.hash import sha256_file
from mdramp.calibration_train import CalibrationMLP
from mdramp.prospective_calibration import SEMANTICS


class JoinedMIC(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();self.head=CalibrationMLP(1571,16);self.head.load_state_dict(checkpoint['state_dict'])
    def forward(self,peptide,pc,means):return self.head(torch.cat([peptide,pc,means],dim=1)).squeeze(-1)


class ProspectiveSequenceScorer(FrozenSequenceScorer):
    """User-authorized adjusted-label prospect, not a paper-prequential score."""
    def __init__(self,config):
        path=Path(config).resolve();self.config=json.loads(path.read_text());base=path.parent
        keys=('esm_checkpoint','scaler','amp_prior','membrane','target_affinity','mic_checkpoint','target_embeddings','target_embedding_metadata','panel_manifest')
        for key in keys:
            if key not in self.config:raise ValueError('Missing prospective artifact: '+key)
            self.config[key]=str((base/self.config[key]).absolute() if key=='esm_checkpoint' else (base/self.config[key]).resolve())
        self.hashes={key:sha256_file(self.config[key]) for key in keys}
        self.fingerprints={key:self._fingerprint(self.config[key]) for key in keys}
        self.cache={};self.details={};self.encoder=None
        checkpoint=torch.load(self.config['mic_checkpoint'],map_location='cpu',weights_only=True)
        if checkpoint.get('task')!='prospective_adjusted_best_mic' or checkpoint.get('target_semantics')!=SEMANTICS or checkpoint.get('target_divisor')!=10 or checkpoint.get('input_dim')!=1571 or checkpoint.get('hidden_dim')!=16 or checkpoint.get('activation')!='GELU':raise ValueError('Explicit adjusted prospective 1571->16->1 checkpoint required')
        if checkpoint.get('historical_reproduction') is not False:raise ValueError('New prospective state must not claim historical reproduction')
        if checkpoint.get('mean')!=[0.]*1571 or checkpoint.get('scale')!=[1.]*1571:raise ValueError('Head must use frozen public descriptor scaling only')
        required=set(keys)-{'mic_checkpoint'}
        if any(checkpoint.get('frozen_artifact_sha256',{}).get(key)!=self.hashes[key] for key in required):raise ValueError('MIC head and frozen expert/panel versions differ')
        self.checkpoint=checkpoint;self._validate_constants()

    def _initialize(self):
        device=self.config.get('device','cpu')
        mps_state=torch.mps.get_rng_state() if device.startswith('mps') else None
        devices=list(range(torch.cuda.device_count())) if device.startswith('cuda') else []
        try:
            with torch.random.fork_rng(devices=devices):self._initialize_impl()
        finally:
            if mps_state is not None:torch.mps.set_rng_state(mps_state)

    def _initialize_impl(self):
        from src.features.esm3_encoder import ESM3Encoder
        from mdramp.prospective_experts import load_experts
        c=self.config;device=c.get('device','cpu');self.verify_artifacts()
        self.scaler,self.prior,self.membrane,self.interaction,self.targets,self.target_ids=load_experts(c,self.hashes,device)
        self.mic=JoinedMIC(self.checkpoint).to(device).eval().requires_grad_(False)
        self.encoder=ESM3Encoder(model_name=c.get('esm_model','esm3-sm-open-v1'),device=device,weights_path=c['esm_checkpoint'])

    def _score(self,sequences):
        import numpy as np
        import pandas as pd
        from mdramp.prospective_experts import score_experts
        from src.features.physicochemical import compute_descriptors
        from src.models.gated_integrator import GateParams,integrate_numpy
        if self.encoder is None:self._initialize()
        device=self.config.get('device','cpu')
        vectors=np.stack([self.encoder.encode_sequence(str(i),seq).global_embedding for i,seq in enumerate(sequences)])
        descriptors=pd.DataFrame([compute_descriptors(seq) for seq in sequences])
        if list(descriptors.columns)!=self.scaler.columns:raise ValueError('Descriptor column mismatch')
        pep=torch.as_tensor(vectors,dtype=torch.float32,device=device)
        pc=torch.as_tensor(self.scaler.transform(descriptors.to_numpy(dtype=np.float32)),device=device)
        with torch.inference_mode():
            evidence=score_experts(pep,pc,self.prior,self.membrane,self.interaction,self.targets,self.target_ids,pair_batch_size=int(self.config.get('pair_batch_size',1024)))
            means=torch.stack([evidence[key] for key in ['mu_prior','mu_mem','mu_int']],dim=1)
            prediction=self.mic(pep,pc,means);activity=torch.sigmoid(5-prediction)
        values={key:value.detach().cpu().numpy() if torch.is_tensor(value) else value for key,value in evidence.items()}
        integration=integrate_numpy(values['mu_prior'],values['mu_mem'],values['mu_int'],GateParams(.5,10.,.3),values['sigma_prior'],values['sigma_mem'],values['sigma_int'],1.)
        frame=pd.DataFrame(dict(sequence=sequences,**values,**integration))
        for name in ('prior','mem','int'):frame['S_'+name]=np.clip(frame['mu_'+name]-frame['sigma_'+name],0,1)
        frame['predicted_adjusted_log2_mic']=prediction.cpu().numpy();frame['A_t']=activity.cpu().numpy();frame['R_t']=frame.p_cons*frame.A_t
        if not np.isfinite(frame.select_dtypes(include='number').to_numpy()).all() or not frame.R_t.between(0,1).all():raise ValueError('Nonfinite prospective score')
        frame['target_semantics']=SEMANTICS;frame['target_divisor']=10.;frame['historical_reproduction']=False;frame['state_id']=self.checkpoint['state_id']
        self.details.update({row['sequence']:row for row in frame.to_dict('records')});self.cache.update(zip(frame.sequence,frame.R_t))
