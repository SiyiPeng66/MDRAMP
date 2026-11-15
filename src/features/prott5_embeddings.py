"""Resumable original ProtT5 per-residue embeddings for TPepPro (CPU float32)."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import zipfile
import numpy as np
import pandas as pd
import torch
from src.utils.hash import sha256_file


def normalize_sequence(sequence):
    sequence=str(sequence).strip().upper()
    if not sequence or re.search('[^ACDEFGHIKLMNPQRSTVWYUZOBX]',sequence):
        raise ValueError('Empty sequence or unsupported amino acid; no truncation/substitute features allowed')
    return re.sub('[UZOB]','X',sequence)


class ProtT5Encoder:
    def __init__(self,model_dir,device='cpu'):
        from transformers import T5EncoderModel,T5Tokenizer
        self.tokenizer=T5Tokenizer.from_pretrained(str(model_dir),do_lower_case=False,local_files_only=True,legacy=True)
        self.model=T5EncoderModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.float32).to(device).float()
        self.model.eval();self.model.requires_grad_(False);self.device=device
    @torch.no_grad()
    def encode(self,sequence):
        sequence=normalize_sequence(sequence)
        tokens=self.tokenizer.batch_encode_plus([' '.join(sequence)],add_special_tokens=True,padding='longest')
        ids=torch.tensor(tokens['input_ids'],device=self.device)
        mask=torch.tensor(tokens['attention_mask'],device=self.device)
        if int(mask[0].sum())!=len(sequence)+1 or ids[0,len(sequence)].item()!=self.tokenizer.eos_token_id:
            raise ValueError('ProtT5 tokenization must contain exactly L residues plus terminal EOS')
        value=self.model(input_ids=ids,attention_mask=mask).last_hidden_state[0,:len(sequence)].cpu().numpy().astype(np.float32)
        validate_array(value,len(sequence))
        return value


def validate_array(value,length):
    if value.shape!=(length,1024) or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('Expected finite float32 Lx1024 ProtT5 residue embeddings')


def atomic_write(path,writer,binary=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb' if binary else 'w') as handle:
            writer(handle);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def atomic_json(path,payload):atomic_write(path,lambda handle:json.dump(payload,handle,indent=2,sort_keys=True),False)


def model_hashes(directory):
    directory=Path(directory)
    required=['config.json','spiece.model','tokenizer_config.json','special_tokens_map.json']
    if not any((directory/name).is_file() for name in ['pytorch_model.bin','model.safetensors']):raise ValueError('Local complete pinned ProtT5 weights required')
    for name in required:
        if not (directory/name).is_file():raise ValueError(f'Missing local model artifact: {name}')
    return {str(path.relative_to(directory)):sha256_file(path) for path in sorted(directory.rglob('*')) if path.is_file() and '.cache' not in path.parts and '.git' not in path.parts}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-csv',required=True);p.add_argument('--id-column',default='record_id');p.add_argument('--sequence-column',default='sequence')
    p.add_argument('--model-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',choices=['cpu'],default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.threads<1:p.error('--threads must be positive')
    output=Path(a.output_dir).resolve();existed=output.exists()
    if existed and not a.resume:raise ValueError('Output exists; use --resume or a new directory')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('ProtT5 output is locked by another process') from exc
        if not a.resume and any(path.name!=".lock" for path in output.iterdir()):raise ValueError("Output became populated; explicit --resume required")
        torch.set_num_threads(a.threads)
        runtime={'python':platform.python_version(),'torch':str(torch.__version__)}
        for name in ['transformers','sentencepiece','numpy']:
            try:runtime[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:runtime[name]=None
        binding={'input_csv':str(Path(a.input_csv).resolve()),'input_sha256':sha256_file(a.input_csv),'model_dir':str(Path(a.model_dir).resolve()),'model_sha256':model_hashes(a.model_dir),'output_dir':str(output),'id_column':a.id_column,'sequence_column':a.sequence_column,'dtype':'float32','device':a.device,'threads':a.threads,'runtime':runtime,'implementation_sha256':sha256_file(__file__),'tokenization':'uppercase; UZOB->X; space-separated residues; add EOS; remove exactly terminal EOS; no truncation'}
        bp=output/'binding.json'
        if bp.exists() and json.loads(bp.read_text())!=binding:raise ValueError('ProtT5 input/model/tokenizer/runtime binding mismatch')
        if not bp.exists():
            if existed and any(path.name!='.lock' for path in output.iterdir()):raise ValueError('Unbound existing output directory')
            atomic_json(bp,binding)
        frame=pd.read_csv(a.input_csv,dtype=str,keep_default_na=False)
        if not {a.id_column,a.sequence_column}.issubset(frame):raise ValueError('Missing ID or sequence column')
        if frame.empty or frame[a.id_column].duplicated().any():raise ValueError('Nonempty unique sequence identifiers required')
        records=[]
        for identifier,sequence in zip(frame[a.id_column],frame[a.sequence_column]):
            if not identifier or identifier in {'.','..'} or '/' in identifier or '\\' in identifier or '\x00' in identifier:raise ValueError('Unsafe NPZ identifier')
            records.append((identifier,normalize_sequence(sequence)))
        bind_hash=sha256_file(bp);encoder=None;cache=output/'cache';cache.mkdir(exist_ok=True);archive=[]
        for number,(identifier,sequence) in enumerate(records,1):
            key=hashlib.sha256(sequence.encode()).hexdigest();path=cache/(key+'.npz');meta=cache/(key+'.json')
            expected={'sequence_sha256':key,'length':len(sequence),'binding_sha256':bind_hash}
            if path.exists() and meta.exists():
                record=json.loads(meta.read_text())
                if any(record.get(k)!=v for k,v in expected.items()) or sha256_file(path)!=record.get('sha256'):raise ValueError(f'Corrupt or foreign cache entry: {path}')
                with np.load(path,allow_pickle=False) as saved:validate_array(saved['embedding'],len(sequence))
            else:
                # An interrupted atomic pair publication is recomputed under this output's binding.
                if encoder is None:encoder=ProtT5Encoder(a.model_dir,a.device)
                value=encoder.encode(sequence);validate_array(value,len(sequence))
                atomic_write(path,lambda handle:np.savez_compressed(handle,embedding=value))
                record={**expected,'sha256':sha256_file(path)};atomic_json(meta,record)
            archive.append({'id':identifier,'sequence_sha256':key,'length':len(sequence),'cache_sha256':record['sha256']})
            if number%10==0 or number==len(records):print(f'ProtT5 cached {number}/{len(records)} sequences',flush=True)
        if sha256_file(a.input_csv)!=binding['input_sha256'] or model_hashes(a.model_dir)!=binding['model_sha256']:raise ValueError('Input/model changed during ProtT5 encoding')
        if sha256_file(__file__) != binding['implementation_sha256'] or json.loads(bp.read_text()) != binding:
            raise ValueError('ProtT5 implementation or binding changed during encoding')
        final=output/'embeddings.npz';manifest=output/'manifest.json'
        if final.exists() and manifest.exists():
            previous=json.loads(manifest.read_text())
            if previous.get('binding_sha256')!=bind_hash or previous.get('output_sha256')!=sha256_file(final):raise ValueError('Completed embedding archive changed')
            print('Completed ProtT5 archive verified',flush=True);return
        def pack(handle):
            with zipfile.ZipFile(handle,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as zf:
                for record in archive:
                    with np.load(cache/(record['sequence_sha256']+'.npz'),allow_pickle=False) as saved:
                        with zf.open(record['id']+'.npy','w',force_zip64=True) as member:np.lib.format.write_array(member,saved['embedding'],allow_pickle=False)
        atomic_write(final,pack)
        atomic_json(manifest,{'schema_version':1,'binding_sha256':bind_hash,'output_sha256':sha256_file(final),'records':archive,'model':'Rostlab/prot_t5_xl_half_uniref50-enc','synthetic_features':False})
        print(f'ProtT5 archive complete: {len(records)} IDs -> {final}',flush=True)


if __name__=='__main__':main()
