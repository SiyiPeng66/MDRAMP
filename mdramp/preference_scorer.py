"""Strict raw-sequence scorer for a separately trained preference MIC head.

The frozen ESM3, expert inference and panel aggregation are inherited unchanged.
Only the MIC head differs, and outputs carry preference-specific semantics.
No per-sequence override, rank target or adjustment policy is read at inference.
"""
from pathlib import Path
import json
import numpy as np
import torch
from mdramp.prospective_scorer import ProspectiveSequenceScorer
from mdramp.preference_head import model_from_checkpoint, TASK
from src.utils.hash import sha256_file

SEMANTICS='explicit_training_preference_not_measured_mic'


class PreferenceSequenceScorer(ProspectiveSequenceScorer):
    def __init__(self,config):
        path=Path(config).resolve();self.config=json.loads(path.read_text());base=path.parent
        if self.config.get('scorer_kind')!='preference_mic_head':
            raise ValueError('Explicit scorer_kind=preference_mic_head required')
        keys=('esm_checkpoint','scaler','amp_prior','membrane','target_affinity','mic_checkpoint','target_embeddings','target_embedding_metadata','panel_manifest')
        for key in keys:
            if key not in self.config:raise ValueError('Missing preference artifact: '+key)
            self.config[key]=str((base/self.config[key]).absolute() if key=='esm_checkpoint' else (base/self.config[key]).resolve())
        self.hashes={key:sha256_file(self.config[key]) for key in keys}
        self.fingerprints={key:self._fingerprint(self.config[key]) for key in keys}
        self.cache={};self.details={};self.encoder=None
        checkpoint=torch.load(self.config['mic_checkpoint'],map_location='cpu',weights_only=True)
        expected={'task':TASK,'target_semantics':SEMANTICS,'target_divisor':None,'input_dim':1571,'hidden_dim':16,'output_dim':1,'activation':'GELU','historical_reproduction':False,'inference_sequence_lookup':False,'inference_rank_override':False}
        if any(k not in checkpoint or checkpoint[k]!=v for k,v in expected.items()):
            raise ValueError('Explicit trained preference 1571->16->1 checkpoint required')
        if checkpoint.get('mean')!=[0.]*1571 or checkpoint.get('scale')!=[1.]*1571:
            raise ValueError('Preference head requires frozen public descriptor scaling')
        frozen={key:self.hashes[key] for key in keys if key!='mic_checkpoint'}
        if checkpoint.get('frozen_artifact_sha256')!=frozen:
            raise ValueError('Preference head and frozen expert/panel versions differ')
        if not checkpoint.get('baseline_checkpoint_sha256') or not checkpoint.get('training_input_sha256'):
            raise ValueError('Preference checkpoint requires training/baseline provenance')
        model_from_checkpoint(checkpoint)
        self.checkpoint=checkpoint;self._validate_constants()

    def _score(self,sequences):
        # Reuse the exact frozen forward path, then replace only output semantics.
        super()._score(sequences)
        for sequence in sequences:
            row=self.details[sequence]
            row['predicted_preference_log2_mic']=row.pop('predicted_adjusted_log2_mic')
            row['predicted_preference_mic_ug_ml']=float(np.exp2(row['predicted_preference_log2_mic']))
            row['target_semantics']=SEMANTICS
            row['target_divisor']=None
            row['inference_sequence_lookup']=False
            row['inference_rank_override']=False

    def validate_artifacts(self):
        self.verify_artifacts()
        return {'state_id':self.checkpoint['state_id'],'task':TASK,'target_semantics':SEMANTICS,'artifact_sha256':dict(self.hashes),'historical_reproduction':False}


def main():
    import argparse
    import pandas as pd
    from mdramp.calibration_train import fresh_output
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--scorer-config',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--threads',type=int,default=2);p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(a.threads)
    frame=pd.read_csv(a.input)
    if 'sequence' not in frame or frame.sequence.isna().any() or frame.sequence.duplicated().any():raise ValueError('Unique sequences required')
    scorer=PreferenceSequenceScorer(a.scorer_config);chunks=[]
    for start in range(0,len(frame),a.batch_size):chunks.append(scorer.score_frame(frame.sequence.iloc[start:start+a.batch_size].tolist()))
    result=pd.concat(chunks,ignore_index=True)
    if 'round' in frame:
        result=result.merge(frame[['sequence','round']],on='sequence',validate='one_to_one')
        result['rank']=pd.Series(pd.NA,index=result.index,dtype='Int64')
        for _,g in result.groupby('round'):
            ids=g.sort_values(['R_t','sequence'],ascending=[False,True]).index
            result.loc[ids,'rank']=np.arange(1,len(ids)+1)
    scorer.verify_artifacts()
    with fresh_output(a.output_dir) as out:
        result.to_csv(out/'scores.csv',index=False)
        (out/'manifest.json').write_text(json.dumps({'task':'raw_sequence_preference_inference','artifact_sha256':scorer.hashes,'input_sha256':sha256_file(a.input),'config_sha256':sha256_file(a.scorer_config),'outputs':{'scores.csv':sha256_file(out/'scores.csv')},'sequence_override':False},indent=2))

if __name__=='__main__':main()
