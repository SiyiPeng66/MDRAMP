"""Bind a newly constructed reference panel to verified real ESM3 artifacts."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.hash import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('panel-manifest','proteins','embeddings','embedding-metadata','esm-checkpoint','embedding-provenance','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();manifest=json.loads(Path(a.panel_manifest).read_text());seq=pd.read_csv(a.proteins);meta=pd.read_csv(a.embedding_metadata)
    if manifest.get('status')!='frozen_new_prospective_panel':raise ValueError('New versioned panel required')
    if manifest['output_sha256']['proteins.csv']!=sha256_file(a.proteins):raise ValueError('Protein source hash mismatch')
    if seq.record_id.duplicated().any() or meta.record_id.duplicated().any() or meta.global_index.duplicated().any():raise ValueError('Duplicate panel IDs/indices')
    if set(seq.record_id)!=set(meta.record_id) or seq.record_id.tolist()!=manifest['record_ids']:raise ValueError('Panel ID mismatch')
    aligned=meta.set_index('record_id').loc[seq.record_id]
    if aligned.sequence.tolist()!=seq.sequence.tolist():raise ValueError('Panel sequence mismatch')
    with np.load(a.embeddings,allow_pickle=False) as data:
        v=data['global_embeddings'];indices=meta.global_index.to_numpy()
        if v.shape!=(len(seq),1536) or not np.isfinite(v).all() or not np.array_equal(np.sort(indices),np.arange(len(seq))):raise ValueError('Invalid panel ESM vectors')
    provenance=json.loads(Path(a.embedding_provenance).read_text())
    if provenance.get('status')!='complete' or provenance.get('encoder')!='ESM3' or provenance.get('binding',{}).get('checkpoint_sha256')!=sha256_file(a.esm_checkpoint) or provenance['binding'].get('input_sha256')!=sha256_file(a.proteins):raise ValueError('ESM provenance mismatch')
    for file in (a.embeddings,a.embedding_metadata):
        if provenance.get('output_sha256',{}).get(Path(file).name)!=sha256_file(file):raise ValueError('ESM output provenance mismatch')
    result={**manifest,'embedding_provenance_sha256':sha256_file(a.embedding_provenance),'embedding_provenance_path':str(Path(a.embedding_provenance).resolve()),'record_ids':meta.record_id.tolist(),'source_manifest_path':str(Path(a.panel_manifest).resolve()),'source_manifest_sha256':sha256_file(a.panel_manifest),'historical_reproduction':False,'artifact_sha256':{'target_embeddings':sha256_file(a.embeddings),'target_embedding_metadata':sha256_file(a.embedding_metadata),'esm_checkpoint':sha256_file(a.esm_checkpoint)}}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2)
    print('Frozen panel embeddings:',len(seq))

if __name__=='__main__':main()
