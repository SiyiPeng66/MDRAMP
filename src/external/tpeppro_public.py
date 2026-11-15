"""Resumable original TPepPro scoring of public structural pairs, without fabricated inputs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import torch
from src.external.tpeppro_runtime import TPepProRuntime
from src.features.prott5_embeddings import atomic_json,atomic_write
from src.utils.hash import sha256_file


@torch.no_grad()
def score_encoded(model,receptor,peptide):
    combined=torch.cat([receptor,peptide],1)
    hidden=torch.relu(model.BatchNorm1d_1(model.fc1(combined)))
    hidden=torch.relu(model.BatchNorm1d_2(model.fc2(hidden)))
    return model.out(hidden).sigmoid().item()


def safe_id(value):
    value=str(value)
    if not value or value in {'.','..'} or '/' in value or '\\' in value or '\x00' in value:raise ValueError('Unsafe record ID')
    return value


def validate_record(identifier,sequences,embeddings,contacts):
    if identifier not in sequences:raise ValueError('sequence_missing')
    if identifier not in embeddings:raise ValueError('embedding_missing')
    path=Path(contacts)/(safe_id(identifier)+'.npz')
    if not path.is_file():raise ValueError('contact_missing')
    with np.load(path,allow_pickle=False) as record:
        sequence=str(record['seq']);contact=record['contact'].copy()
    expected=sequences[identifier];embedding=np.asarray(embeddings[identifier],dtype=np.float32)
    if sequence!=expected:raise ValueError('contact_sequence_mismatch')
    if embedding.shape!=(len(expected),1024) or contact.shape!=(len(expected),len(expected)):raise ValueError('input_shape_mismatch')
    if not np.isfinite(embedding).all() or not np.isfinite(contact).all():raise ValueError('nonfinite_input')
    return embedding,contact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['pairs','sequences','embeddings','checkpoint','contact-dir','structure-manifest','embedding-manifest','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();out=Path(a.output_dir).resolve()
    if out.exists() and not a.resume:raise ValueError('Output exists; explicit resume required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise ValueError('Teacher output locked') from e
        if not a.resume and any(x.name!='.lock' for x in out.iterdir()):raise ValueError('Output became populated')
        torch.set_num_threads(a.threads)
        pairs=pd.read_csv(a.pairs,dtype=str,keep_default_na=False);seqs=pd.read_csv(a.sequences,dtype=str,keep_default_na=False)
        if not {'pair_id','receptor','peptide'}.issubset(pairs) or pairs.pair_id.duplicated().any():raise ValueError('Unique pair_id/receptor/peptide required')
        if not {'record_id','sequence'}.issubset(seqs) or seqs.record_id.duplicated().any():raise ValueError('Unique record_id sequence table required')
        sequences=dict(zip(seqs.record_id,seqs.sequence));ids=sorted(set(pairs.receptor)|set(pairs.peptide))
        contacts={identifier:(sha256_file(Path(a.contact_dir)/(safe_id(identifier)+'.npz')) if (Path(a.contact_dir)/(safe_id(identifier)+'.npz')).is_file() else None) for identifier in ids}
        files={name:{'path':str(Path(getattr(a,name)).resolve()),'sha256':sha256_file(getattr(a,name))} for name in ['pairs','sequences','embeddings','checkpoint','structure_manifest','embedding_manifest']}
        structures=json.loads(Path(a.structure_manifest).read_text())
        if structures.get('status')!='complete':raise ValueError('Completed verified structural inputs required')
        for name,key in [('teacher_pairs.csv','pairs'),('teacher_sequences.csv','sequences')]:
            if structures.get('output_sha256',{}).get(name)!=files[key]['sha256']:raise ValueError('Structure sequence/pair table provenance mismatch')
        for identifier,digest in contacts.items():
            record_path=Path(a.structure_manifest).parent/'records'/(safe_id(identifier)+'.json')
            if not record_path.is_file():raise ValueError('Missing individual structure provenance')
            record=json.loads(record_path.read_text())
            if record.get('status')!='complete' or record.get('contact_sha256')!=digest:raise ValueError('Contact structure provenance mismatch')
        prott5=json.loads(Path(a.embedding_manifest).read_text())
        if prott5.get('output_sha256')!=files['embeddings']['sha256']:raise ValueError('ProtT5 archive provenance mismatch')
        t5records=prott5.get('records',[])
        if prott5.get('synthetic_features') is not False or len({r['id'] for r in t5records})!=len(t5records):raise ValueError('Invalid real ProtT5 manifest')
        t5map={record['id']:record for record in t5records}
        for identifier in ids:
            expected=hashlib.sha256(sequences.get(identifier,'').encode()).hexdigest()
            if identifier not in t5map or t5map[identifier].get('sequence_sha256')!=expected:raise ValueError('ProtT5 sequence provenance mismatch')
        binding={'files':files,'contacts':contacts,'contact_dir':str(Path(a.contact_dir).resolve()),'output_dir':str(out),'threads':a.threads,'runtime':{'python':platform.python_version(),'torch':str(torch.__version__)},'code_sha256':{str(path):sha256_file(path) for path in [Path(__file__),Path(__file__).with_name('tpeppro_runtime.py'),Path(__file__).parents[1]/'features/prott5_embeddings.py']}}
        bp=out/'binding.json'
        if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('Teacher resume binding mismatch')
        if not bp.exists():atomic_json(bp,binding)
        binding_sha=sha256_file(bp)
        completed=out/'manifest.json'
        if completed.exists():
            previous=json.loads(completed.read_text())
            if previous.get('binding_sha256')!=binding_sha or sha256_file(out/'teacher_scores.csv')!=previous.get('scores',{}).get('sha256'):raise ValueError('Completed teacher output changed')
            print('Completed teacher scores verified',flush=True);return
        model=None;vectors={};invalid={};results=[]
        cache=out/'pair_cache';cache.mkdir(exist_ok=True);vecdir=out/'vector_cache';vecdir.mkdir(exist_ok=True)
        with np.load(a.embeddings,allow_pickle=False) as embeddings:
            def get_vector(identifier):
                nonlocal model
                if identifier in invalid:raise ValueError(invalid[identifier])
                if identifier in vectors:return vectors[identifier]
                try:embedding,contact=validate_record(identifier,sequences,embeddings,a.contact_dir)
                except (ValueError,KeyError,OSError) as e:invalid[identifier]=str(e);raise ValueError(str(e)) from e
                key=hashlib.sha256(identifier.encode()).hexdigest();path=vecdir/(key+'.npz');meta=vecdir/(key+'.json')
                if path.exists() and meta.exists():
                    info=json.loads(meta.read_text())
                    if info.get('binding_sha256')!=binding_sha or info.get('sha256')!=sha256_file(path):raise RuntimeError('Corrupt vector cache')
                    with np.load(path,allow_pickle=False) as saved:vector=torch.from_numpy(saved['vector'].copy())
                else:
                    if model is None:model=TPepProRuntime(a.checkpoint)
                    with torch.no_grad():vector=model.encode(embedding,contact).detach().cpu()
                    atomic_write(path,lambda handle:np.savez_compressed(handle,vector=vector.numpy()))
                    atomic_json(meta,{'binding_sha256':binding_sha,'sha256':sha256_file(path)})
                if vector.shape!=(1,128) or not torch.isfinite(vector).all():raise RuntimeError('Invalid encoded vector')
                vectors[identifier]=vector;return vector
            for number,row in enumerate(pairs.itertuples(index=False),1):
                key=hashlib.sha256(row.pair_id.encode()).hexdigest();path=cache/(key+'.json');digest=cache/(key+'.sha256')
                if path.exists() and digest.exists():
                    if sha256_file(path)!=digest.read_text().strip():raise ValueError('Corrupt pair cache')
                    result=json.loads(path.read_text())
                    if result.pop('binding_sha256',None)!=binding_sha or result['pair_id']!=row.pair_id:raise ValueError('Foreign pair cache')
                else:
                    result={'pair_id':row.pair_id,'receptor':row.receptor,'peptide':row.peptide,'s_aff_pair':None,'status':'invalid_input','reason':''}
                    try:receptor=get_vector(row.receptor);peptide=get_vector(row.peptide)
                    except ValueError as e:result['reason']=str(e)
                    else:
                        if model is None:model=TPepProRuntime(a.checkpoint)
                        score=score_encoded(model,receptor,peptide)
                        if not np.isfinite(score) or not 0<=score<=1:raise RuntimeError('Invalid TPepPro model output')
                        result.update(s_aff_pair=score,status='scored')
                    atomic_json(path,{**result,'binding_sha256':binding_sha});atomic_write(digest,lambda handle:handle.write(sha256_file(path)),False)
                results.append(result)
                if number%25==0 or number==len(pairs):print(f'TPepPro pairs {number}/{len(pairs)} unique_vectors={len(vectors)}',flush=True)
        for name,item in files.items():
            if sha256_file(item['path'])!=item['sha256']:raise ValueError('Teacher input changed during inference')
        for identifier,digest in contacts.items():
            path=Path(a.contact_dir)/(identifier+'.npz')
            if (sha256_file(path) if path.is_file() else None)!=digest:raise ValueError('Contact changed during inference')
        if any(sha256_file(path) != value for path, value in binding['code_sha256'].items()):
            raise ValueError('TPepPro implementation changed during inference')
        if json.loads(bp.read_text()) != binding:
            raise ValueError('TPepPro binding changed during inference')
        scores=out/'teacher_scores.csv';atomic_write(scores,lambda handle:pd.DataFrame(results).to_csv(handle,index=False),False)
        atomic_json(out/'manifest.json',{'schema_version':1,'binding_sha256':binding_sha,'inputs':files,'scores':{'path':str(scores),'sha256':sha256_file(scores)},'counts':pd.Series([r['status'] for r in results]).value_counts().to_dict(),'contact_sha256':contacts,'runtime':'original TPepPro weights and exact eval-mode cached encode/head operations'})


if __name__=='__main__':main()
