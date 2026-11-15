"""Frozen complete-sequence scoring with retained models and batched target pairs."""
from pathlib import Path
import json
import pandas as pd
from src.utils.hash import sha256_file


class FrozenSequenceScorer:
    """Config keys pin checkpoints, scaler, target panel embeddings and gate parameters.

    Scores are cached per exact sequence for one immutable scorer instance.
    No experimental labels are read and no scoring parameter receives gradients.
    """
    def __init__(self, config):
        if isinstance(config,(str,Path)):
            config_path=Path(config).resolve(); self.config=json.loads(config_path.read_text())
            base=Path(self.config.pop('path_base',config_path.parent))
            if not base.is_absolute(): base=config_path.parent/base
            for key in ('esm_checkpoint','scaler','amp_prior','membrane','target_affinity','mic_checkpoint','target_embeddings','target_embedding_metadata','state_manifest','panel_manifest'):
                if key in self.config and not Path(self.config[key]).is_absolute(): self.config[key]=str((base/self.config[key]).absolute())
        else: self.config=dict(config)
        self.cache = {}
        self.details = {}
        self.encoder = None
        required = ('esm_checkpoint','scaler', 'amp_prior', 'membrane', 'target_affinity', 'mic_checkpoint',
                    'target_embeddings', 'target_embedding_metadata', 'state_manifest', 'panel_manifest')
        for key in required:
            if key not in self.config or not Path(self.config[key]).is_file():
                raise ValueError(f'Missing frozen scorer artifact: {key}')

        self._validate_constants()
        self.hashes = {key: sha256_file(self.config[key]) for key in required}
        self.fingerprints = {key: self._fingerprint(self.config[key]) for key in required}

    def _validate_constants(self):
        for key, expected in {'aggregation_coefficient':.8, 'kappa':1., 'tau':.5,
                              'gamma':10., 'lambda_syn':.3}.items():
            if float(self.config.get(key, expected)) != expected:
                raise ValueError(f'V6 generation requires fixed {key}={expected}')

    @staticmethod
    def _validate_expert_training(key, metadata):
        expected = {'epochs':200, 'batch_size':32 if key == 'target_affinity' else 64,
                    'learning_rate':5e-5 if key == 'target_affinity' else 1e-4}
        actual = metadata.get('training_config', {})
        if any(actual.get(name) != value for name,value in expected.items()):
            raise ValueError(f'V6 S12 training provenance missing or incompatible for {key}: expected {expected}')

    def validate_artifacts(self):
        """CPU-only strict preflight; never initialize or execute ESM3."""
        import numpy as np
        import torch
        from src.features.scaler import StateFeatureScaler
        from src.features.physicochemical import compute_descriptors
        from src.features.embedding_store import EmbeddingStore
        from src.loop.state_manifest import load_manifest
        from src.train.train_mic_head import FROZEN_ARTIFACTS
        self._validate_constants()
        self.verify_artifacts()
        c = self.config
        state = load_manifest(c['state_manifest']); state.assert_paper_policy()
        mic = torch.load(c['mic_checkpoint'], map_location='cpu', weights_only=True)
        meta = mic.get('metadata', {})
        if (meta.get('paper_prequential_validated') is not True or
            meta.get('architecture') != '1571->16->1' or
            (meta.get('peptide_dim'), meta.get('physchem_dim'), meta.get('expert_dim')) != (1536,32,3) or
            meta.get('target_semantics') != 'measured_mic' or meta.get('target_definition') != 'panel_best_mic' or
            meta.get('target_divisor') != 1):
            raise ValueError('V6 requires a measured panel-best MIC checkpoint, 1571->16->1; retrospective or scaled heads are incompatible')
        for key in ('state_name','visible_round','data_cutoff'):
            if meta.get(key) != getattr(state,key): raise ValueError(f'MIC/state mismatch: {key}')
        if meta.get('input_hashes',{}).get('state_manifest') != self.hashes['state_manifest']:
            raise ValueError('MIC checkpoint is not bound to this state manifest')
        for key in FROZEN_ARTIFACTS:
            if meta.get('frozen_artifact_hashes',{}).get(key) != self.hashes[key] or state.input_hashes.get(key) != self.hashes[key]:
                raise ValueError(f'Frozen MIC/expert/panel binding mismatch: {key}')
        scaler = StateFeatureScaler.load(c['scaler'])
        if scaler.columns != list(compute_descriptors('ACDEFGHIKLMNPQRSTVWY')) or not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.scale).all() or not (scaler.scale > 0).all():
            raise ValueError('Invalid frozen 32-descriptor scaler')
        from src.models.mic_regressor import MICRegressionHead
        MICRegressionHead().load_state_dict(mic['state_dict'])
        expected = {'amp_prior':('PaperAMPPriorExpert',{'input_dim':1568}),
                    'membrane':('FeaturewiseGatedMembraneExpert',{'esm_dim':1536,'physchem_dim':32}),
                    'target_affinity':('ProjectedPairAffinityExpert',{'peptide_dim':1536,'target_dim':1536,'physchem_dim':32})}
        for key,(architecture, dimensions) in expected.items():
            expert = torch.load(c[key],map_location='cpu',weights_only=True)['metadata']
            self._validate_expert_training(key, expert)
            if expert.get('architecture') != architecture or expert.get('seeds') != [11,23,37,51,73] or any(expert.get(k)!=v for k,v in dimensions.items()):
                raise ValueError(f'Incompatible V6 expert checkpoint: {key}')
        store = EmbeddingStore.load(c['target_embeddings'],c['target_embedding_metadata'])
        try:
            frame = store.metadata
            if not {'record_id','global_index','sequence'}.issubset(frame): raise ValueError('Panel metadata requires target sequence provenance')
            index = pd.to_numeric(frame.global_index,errors='raise').to_numpy()
            if (len(frame)<5 or frame.record_id.isna().any() or frame.record_id.astype(str).duplicated().any() or
                frame.sequence.isna().any() or not frame.sequence.astype(str).str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+').all() or
                not np.isfinite(index).all() or not np.equal(index,index.astype(int)).all() or len(set(index))!=len(index) or
                store.global_embeddings.ndim!=2 or store.global_embeddings.shape[1]!=1536 or
                (index<0).any() or (index>=len(store.global_embeddings)).any() or not np.isfinite(store.global_embeddings).all()):
                raise ValueError('Invalid frozen protein panel metadata/embeddings')
            panel = json.loads(Path(c['panel_manifest']).read_text())
            if panel.get('frozen_before_r0') is not True or panel.get('record_ids') != frame.record_id.astype(str).tolist():
                raise ValueError('Archived pre-R0 protein panel declaration required; do not fabricate historical provenance')
            for key in ('esm_checkpoint','target_embeddings','target_embedding_metadata'):
                if panel.get('artifact_sha256',{}).get(key) != self.hashes[key]: raise ValueError(f'Panel provenance mismatch: {key}')
            frozen_at = pd.Timestamp(panel.get('frozen_at'))
            r0_cutoff = pd.Timestamp(panel.get('r0_cutoff'))
            if pd.isna(frozen_at) or pd.isna(r0_cutoff) or frozen_at.tzinfo is None or r0_cutoff.tzinfo is None or frozen_at > r0_cutoff or r0_cutoff > pd.Timestamp(state.data_cutoff):
                raise ValueError('Panel requires a valid historical freeze and R0 boundary')
        finally:
            store.npz.close()
        self.verify_artifacts()
        return {'state_name':state.state_name,'architecture':'1571->16->1','artifact_sha256':dict(self.hashes),
                'historical_authenticity':'Requires independently preserved release ledger and pre-R0 panel archive; hash checks alone do not establish historical truth.'}

    @staticmethod
    def _fingerprint(path):
        stat = Path(path).stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def verify_artifacts(self):
        if any(sha256_file(self.config[k]) != v for k,v in self.hashes.items()):
            raise ValueError("Frozen scorer artifact changed")

    def __call__(self, sequences):
        self._validate_constants()
        if any(self._fingerprint(self.config[k]) != v for k,v in self.fingerprints.items()):
            raise ValueError("Frozen scorer artifact changed")
        if isinstance(sequences, str):
            return self([sequences])[0]
        missing = list(dict.fromkeys(s for s in sequences if s not in self.cache))
        if missing:
            self._score(missing)
        return [self.cache[s] for s in sequences]

    def score_frame(self, sequences):
        self(sequences)
        return pd.DataFrame([self.details[s] for s in sequences])

    def _initialize(self):
        import torch
        from src.features.esm3_encoder import ESM3Encoder
        from src.features.scaler import StateFeatureScaler
        from src.features.embedding_store import EmbeddingStore
        from src.score.score_candidates import load_ensemble
        from src.models.ensemble import ExpertEnsemble
        from src.models.expert_modules import ProjectedPairAffinityExpert
        from src.models.mic_regressor import MICRegressionHead
        self.validate_artifacts()
        c=self.config; device=c.get('device','cpu')
        self.encoder=ESM3Encoder(model_name=c.get('esm_model','esm3-sm-open-v1'),device=device,weights_path=c['esm_checkpoint'])
        self.scaler=StateFeatureScaler.load(c['scaler'])
        self.prior,_=load_ensemble(c['amp_prior'],'PaperAMPPriorExpert')
        self.membrane,_=load_ensemble(c['membrane'],'FeaturewiseGatedMembraneExpert')
        ckpt=torch.load(c['target_affinity'],map_location='cpu',weights_only=True); meta=ckpt['metadata']
        if meta.get('architecture')!='ProjectedPairAffinityExpert' or len(meta.get('seeds',[]))!=5: raise ValueError('Five-member projected pair checkpoint required')
        self.interaction=ExpertEnsemble(lambda:ProjectedPairAffinityExpert(meta['peptide_dim'],meta['target_dim'],meta['physchem_dim']),seeds=meta['seeds'])
        self.interaction.load_state_dict(ckpt['state_dict'])
        ckpt=torch.load(c['mic_checkpoint'],map_location='cpu',weights_only=True); meta=ckpt['metadata']
        if not meta.get('paper_prequential_validated'): raise ValueError('Validated MIC checkpoint required')
        self.mic=MICRegressionHead(meta['peptide_dim'],meta['physchem_dim']);self.mic.load_state_dict(ckpt['state_dict'])
        for model in (self.prior,self.membrane,self.interaction,self.mic): model.to(device).eval().requires_grad_(False)
        store=EmbeddingStore.load(c['target_embeddings'],c['target_embedding_metadata'])
        self.target_ids=store.metadata.record_id.astype(str).tolist()
        if len(self.target_ids)<5 or len(set(self.target_ids))!=len(self.target_ids): raise ValueError('Panel requires five or more unique targets')
        self.targets=torch.as_tensor(store.global_by_record_id(self.target_ids),device=device)
        store.npz.close()

    def _score(self, sequences):
        import torch
        import numpy as np
        from src.features.physicochemical import compute_descriptors
        from src.models.gated_integrator import GateParams, integrate_numpy
        from src.models.mic_regressor import activity_priority
        if self.encoder is None: self._initialize()
        c=self.config; device=c.get('device','cpu'); chunk=int(c.get('pair_batch_size',1024))
        if chunk<1: raise ValueError('pair_batch_size must be positive')
        embeddings=np.stack([self.encoder.encode_sequence(str(i),seq).global_embedding for i,seq in enumerate(sequences)])
        descriptors=pd.DataFrame([compute_descriptors(seq) for seq in sequences])
        if list(descriptors.columns)!=self.scaler.columns: raise ValueError('Descriptor order differs from frozen scaler')
        pc=torch.as_tensor(self.scaler.transform(descriptors.to_numpy(dtype=np.float32)),device=device)
        pep=torch.as_tensor(embeddings,device=device)
        with torch.inference_mode():
            prior=self.prior(torch.cat([pep,pc],1)); membrane=self.membrane(pep,pc)
            all_members=[]; best_ids=[]; best_scores=[]
            coefficient=float(c.get('aggregation_coefficient',.8))
            if not 0<=coefficient<=1: raise ValueError('Invalid interaction coefficient')
            for i in range(len(sequences)):
                member_values=[]; pair_means=[]
                for member in self.interaction.members:
                    logits=[]
                    for start in range(0,len(self.targets),chunk):
                        target=self.targets[start:start+chunk]; n=len(target)
                        logits.append(member.pair_logits(pep[i:i+1].expand(n,-1),target,pc[i:i+1].expand(n,-1)))
                    logits=torch.cat(logits); probabilities=logits.sigmoid();pair_means.append(probabilities)
                    top=logits.topk(5).values; q=top.sigmoid()
                    member_values.append(coefficient*(top.softmax(0)*q).sum()+(1-coefficient)*(1-(1-q).prod()))
                mean=torch.stack(pair_means).mean(0); index=mean.argmax().item()
                best_ids.append(self.target_ids[index]);best_scores.append(mean[index].item())
                all_members.append(torch.stack(member_values))
            members=torch.stack(all_members,1); interaction_mean=members.mean(0);interaction_std=members.std(0,unbiased=True)
            means=torch.stack([prior['mean'],membrane['mean'],interaction_mean],1)
            prediction=self.mic(pep,pc,means);activity=activity_priority(prediction)
        array=lambda x:x.detach().cpu().numpy()
        pm,ps,mm,ms,im,ist=map(array,(prior['mean'],prior['std'],membrane['mean'],membrane['std'],interaction_mean,interaction_std))
        kappa=float(c.get('kappa',1.))
        integration=integrate_numpy(pm,mm,im,GateParams(c.get('tau',.5),c.get('gamma',10.),c.get('lambda_syn',.3)),ps,ms,ist,kappa)
        scores=pd.DataFrame(dict(sequence=sequences,mu_prior=pm,sigma_prior=ps,mu_mem=mm,sigma_mem=ms,mu_int=im,sigma_int=ist,
            S_prior=np.clip(pm-kappa*ps,0,1),S_mem=np.clip(mm-kappa*ms,0,1),S_int=np.clip(im-kappa*ist,0,1),
            predicted_log2_mic=array(prediction),A_t=array(activity),top_target_id=best_ids,top_target_score=best_scores,**integration))
        scores['R_t']=scores.p_cons*scores.A_t
        if not np.isfinite(scores.select_dtypes(include='number').to_numpy()).all() or not scores.R_t.between(0,1).all(): raise ValueError('Invalid scoring output')
        self.details.update({row['sequence']:row for row in scores.to_dict('records')})
        self.cache.update(zip(scores.sequence,scores.R_t))
