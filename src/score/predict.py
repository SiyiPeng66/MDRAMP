"""Formal CSV/FASTA batch prediction using frozen MDRAMP checkpoint bundles."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from src.score.frozen_scorer import FrozenSequenceScorer
from src.utils.hash import sha256_file

ARTIFACT_KEYS=('esm_checkpoint','scaler','amp_prior','membrane','target_affinity','mic_checkpoint','target_embeddings','target_embedding_metadata')

def load_candidates(path,id_column='peptide_id',sequence_column='sequence'):
    path=Path(path)
    if path.suffix.lower() in ('.fa','.fasta','.faa','.fas'):
        rows=[]; identifier=None; chunks=[]
        def flush():
            if identifier is not None:
                if not chunks: raise ValueError('Empty FASTA record: '+identifier)
                rows.append({'peptide_id':identifier,'sequence':''.join(chunks)})
        for line in path.read_text().splitlines():
            line=line.strip()
            if not line: continue
            if line.startswith('>'):
                flush(); identifier=line[1:].split()[0] if line[1:].strip() else ''; chunks=[]
            else:
                if identifier is None: raise ValueError('Sequence before FASTA header')
                chunks.append(line)
        flush(); frame=pd.DataFrame(rows,columns=['peptide_id','sequence'])
    else:
        frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        if not {id_column,sequence_column}.issubset(frame): raise ValueError('CSV requires explicit ID and sequence columns')
        frame=frame[[id_column,sequence_column]].rename(columns={id_column:'peptide_id',sequence_column:'sequence'})
    if frame.empty: raise ValueError('Candidate input is empty')
    frame['sequence']=frame.sequence.str.strip().str.upper()
    if frame.peptide_id.str.strip().eq('').any() or frame.peptide_id.duplicated().any(): raise ValueError('Candidate IDs must be nonempty and unique')
    valid=frame.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10,50}')
    if not valid.all(): raise ValueError('Candidates require 10–50 standard amino acids; invalid IDs: '+','.join(frame.loc[~valid,'peptide_id'].head(10)))
    return frame

def load_config(path,device=None):
    path=Path(path).resolve(); config=json.loads(path.read_text())
    # Existing project configs are relative to the invocation directory. Declare
    # path_base explicitly for portable bundles; absent means config directory.
    base=Path(config.pop('path_base',path.parent))
    if not base.is_absolute(): base=path.parent/base
    for key in ARTIFACT_KEYS:
        if key not in config: raise ValueError('Missing scorer config key: '+key)
        value=Path(config[key]); config[key]=str(value if value.is_absolute() else (base/value).absolute())
    if device is not None: config['device']=device
    return config

def predict(frame,scorer,batch_size=128):
    if batch_size<1: raise ValueError('Batch size must be positive')
    parts=[]
    for start in range(0,len(frame),batch_size):
        batch=frame.iloc[start:start+batch_size]
        scores=scorer.score_frame(batch.sequence.tolist()).reset_index(drop=True)
        if len(scores)!=len(batch) or scores.sequence.tolist()!=batch.sequence.tolist(): raise ValueError('Prediction row alignment failed')
        scores.insert(0,'peptide_id',batch.peptide_id.tolist()); parts.append(scores)
    result=pd.concat(parts,ignore_index=True)
    if not np.isfinite(result.R_t.to_numpy()).all(): raise ValueError('Nonfinite ranking score')
    result=result.sort_values(['R_t','peptide_id'],ascending=[False,True],kind='stable').reset_index(drop=True)
    result.insert(0,'rank',np.arange(1,len(result)+1)); return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--scorer-config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--id-column',default='peptide_id');p.add_argument('--sequence-column',default='sequence');p.add_argument('--device')
    a=p.parse_args(argv)
    if a.batch_size<1: p.error('--batch-size must be positive')
    output=Path(a.output).resolve(); input_path=Path(a.input).resolve()
    if output in (input_path,Path(a.scorer_config).resolve()): p.error('Output must not overwrite input/config')
    input_hash = sha256_file(input_path)
    config_hash = sha256_file(a.scorer_config)
    frame=load_candidates(input_path,a.id_column,a.sequence_column); config=load_config(a.scorer_config,a.device)
    if str(output) in [config[k] for k in ARTIFACT_KEYS]: p.error('Output must not overwrite model artifacts')
    scorer=FrozenSequenceScorer(config); result=predict(frame,scorer,a.batch_size)
    scorer.verify_artifacts()
    if sha256_file(input_path) != input_hash or sha256_file(a.scorer_config) != config_hash:
        raise ValueError("Input/config changed during prediction; output not published")
    output.parent.mkdir(parents=True,exist_ok=True)
    # Publish output only after all batches succeed.
    with tempfile.NamedTemporaryFile(mode='w',dir=output.parent,suffix='.csv',delete=False) as handle:
        temporary=Path(handle.name);result.to_csv(handle,index=False)
    temporary.replace(output)
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':input_hash,'config_sha256':config_hash,'output_sha256':sha256_file(output),'artifact_sha256':scorer.hashes,'resolved_config':config,'candidate_count':len(frame),'unique_sequences':frame.sequence.nunique(),'arguments':vars(a),'tie_rule':'descending R_t, ascending peptide_id','score_interpretation':'priority signals, not calibrated activity probabilities'}
    Path(str(output)+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Predicted {len(frame)} candidates -> {output}')
if __name__=='__main__':main()
