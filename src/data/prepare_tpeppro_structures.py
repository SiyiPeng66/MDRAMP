"""Fetch verified RCSB chains for TPepPro background-screening inputs.

No sequence/structure substitution, alignment, truncation or synthetic contact
maps are permitted. This prepares teacher inputs; it does not run the teacher.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import Bio
from src.data.prepare_pairs import read_dictionary
from src.utils.hash import sha256_file

AA={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}


def atomic_json(path,value):
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def contact_from_chain(chain,expected):
    residues=[residue for residue in chain if residue.id[0]==' ' and residue.resname in AA]
    sequence=''.join(AA[r.resname] for r in residues)
    if sequence!=expected:
        raise ValueError(f'observed_sequence_mismatch expected_length={len(expected)} observed_length={len(sequence)} observed_sha256={hashlib.sha256(sequence.encode()).hexdigest()}')
    if not residues:raise ValueError('empty_observed_chain')
    if any('CA' not in residue for residue in residues):raise ValueError('missing_CA_atom')
    coordinates=np.asarray([residue['CA'].coord for residue in residues],dtype=np.float64)
    if not np.isfinite(coordinates).all():raise ValueError('nonfinite_CA_coordinates')
    contacts=np.empty((len(residues),len(residues)),dtype=np.uint8)
    for start in range(0,len(residues),128):
        delta=coordinates[start:start+128,None,:]-coordinates[None,:,:]
        contacts[start:start+128]=(np.sum(delta*delta,axis=2)<64.).astype(np.uint8)
    np.fill_diagonal(contacts,0)
    evidence={'observed_sequence':sequence,'residue_ids':[str(r.id) for r in residues],
              'CA_altlocs':[str(r['CA'].get_altloc()) for r in residues],
              'CA_occupancies':[r['CA'].get_occupancy() for r in residues],
              'model_index':0,'chain_id':str(chain.id),'contact_rule':'CA Euclidean distance <8 Angstrom; diagonal zero'}
    return contacts,evidence


def parse_chain(path,pdb_id,chain_id,expected):
    from Bio.PDB import FastMMCIFParser
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        structure=FastMMCIFParser(QUIET=True,auth_chains=True,auth_residues=True).get_structure(pdb_id,stream)
    model=next(structure.get_models())
    if chain_id not in model:raise ValueError('author_chain_not_found')
    return contact_from_chain(model[chain_id],expected)


class StructureCache:
    def __init__(self,path,retry_failed=False,timeout=30.,attempts=2):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.retry_failed=retry_failed;self.timeout=timeout;self.attempts=attempts
        self.guard=threading.Lock();self.locks={};self.rate_lock=threading.Lock();self.next_request=0.
    def fetch(self,pdb_id):
        pdb_id=pdb_id.lower()
        if not re.fullmatch(r'[0-9][a-z0-9]{3}',pdb_id):raise ValueError('unsupported_PDB_identifier')
        with self.guard:lock=self.locks.setdefault(pdb_id,threading.Lock())
        with lock:
            path=self.path/(pdb_id+'.cif.gz');receipt=self.path/(pdb_id+'.json')
            if receipt.exists():
                previous=json.loads(receipt.read_text())
                if previous['status']=='complete':
                    if not path.is_file() or sha256_file(path)!=previous['sha256']:raise ValueError('download_cache_hash_mismatch')
                    return path,previous
                if not self.retry_failed:raise ValueError('cached_download_failure: '+previous.get('error',''))
            url=f'https://files.rcsb.org/download/{pdb_id}.cif.gz';error=''
            for attempt in range(self.attempts):
                with self.rate_lock:
                    wait=max(0.,self.next_request-time.monotonic())
                    if wait:time.sleep(wait)
                    self.next_request=time.monotonic()+.25
                try:
                    req=Request(url,headers={'User-Agent':'MDRAMP-public-structure-preparation/1.0'})
                    with urlopen(req,timeout=self.timeout) as response:
                        fd,temp=tempfile.mkstemp(prefix='.'+pdb_id,dir=self.path)
                        try:
                            size=0
                            with os.fdopen(fd,'wb') as stream:
                                while True:
                                    block=response.read(1024*1024)
                                    if not block:break
                                    size+=len(block)
                                    if size>128*1024*1024:raise ValueError('compressed_structure_exceeds_128MiB_limit')
                                    stream.write(block)
                                stream.flush();os.fsync(stream.fileno())
                            with gzip.open(temp,'rb') as check:
                                if not check.read(512).lstrip().startswith(b'data_'):raise ValueError('response_is_not_mmcif')
                                while check.read(1024*1024):pass  # Validate gzip CRC before caching success.
                            os.replace(temp,path)
                        finally:
                            if os.path.exists(temp):os.unlink(temp)
                        record={'status':'complete','url':url,'sha256':sha256_file(path),'bytes':size,
                                'downloaded_at':datetime.now(timezone.utc).isoformat(),'etag':response.headers.get('ETag'),
                                'last_modified':response.headers.get('Last-Modified')}
                    atomic_json(receipt,record);return path,record
                except (HTTPError,URLError,TimeoutError,OSError,ValueError) as exc:
                    error=f'{type(exc).__name__}: {exc}'
                    if isinstance(exc,HTTPError) and exc.code in {400,404,410}:break
                    if attempt+1<self.attempts:time.sleep(min(2**attempt,4))
            atomic_json(receipt,{'status':'failed','url':url,'error':error,'attempted_at':datetime.now(timezone.utc).isoformat()})
            raise ValueError('download_failed: '+error)


def _save_contact(path,sequence,contact):
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,seq=sequence,contact=contact);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def prepare(root,output_dir,cache_dir,workers=4,resume=False,retry_failed=False,limit_pairs=None):
    if not 1<=workers<=4:raise ValueError('workers must be between 1 and 4')
    if limit_pairs is not None and limit_pairs<1:raise ValueError('limit-pairs must be positive')
    root=Path(root).resolve();output=Path(output_dir).absolute();cache=StructureCache(cache_dir,retry_failed=retry_failed)
    source=root/'outputs/public_pairs/all_pairs.csv'
    dictionary=root/'otherModels/TPepPro/data/receptor(14374)-peptide(9594)_dictionary.xlsx'
    parser_path = Path(__file__).with_name('prepare_pairs.py')
    hashes={'pairs':sha256_file(source),'dictionary':sha256_file(dictionary),
            'implementation':sha256_file(__file__),'dictionary_parser':sha256_file(parser_path)}
    pairs=pd.read_csv(source,dtype={'source_peptide_ids':str,'source_target_ids':str})
    pairs=pairs[pairs.label_type.eq('background')&pairs.split.isin(['train','validation','test'])].sort_values('pair_id').reset_index(drop=True)
    if limit_pairs is not None:pairs=pairs.head(limit_pairs)
    rows=read_dictionary(dictionary);sequence_ids=rows.groupby('sequence').source_id.agg(lambda values:sorted(set(values))).to_dict()
    nodes={}
    for row in pairs.itertuples():
        for record_id,sequence,column in [(row.peptide_id,row.sequence,'source_peptide_ids'),(row.target_id,row.target_sequence,'source_target_ids')]:
            candidates=sequence_ids.get(sequence,[])
            if record_id in nodes and nodes[record_id]['sequence']!=sequence:raise ValueError('record_id_sequence_conflict')
            nodes[record_id]={'record_id':record_id,'sequence':sequence,'source_ids':candidates}
    binding={'input_sha256':hashes,'limit_pairs':limit_pairs,'record_count':len(nodes),
             'runtime':{'biopython':Bio.__version__,'numpy':np.__version__},
             'contact_rule':'first model author-chain, observed standard AA exact sequence, all CA required, distance<8A, diagonal=0',
             'teacher_screening':'not_yet_performed'}
    if output.exists() and not resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (output/'.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Structure preparation already running') from exc
        manifest_path=output/'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()).get('binding')!=binding:raise ValueError('Preparation input/configuration changed; use a new output directory')
        elif resume:raise ValueError('Cannot resume without bound manifest')
        for name in ['contacts','records']:(output/name).mkdir(exist_ok=True)
        manifest={'schema_version':1,'status':'running','binding':binding,'total_background_pairs':len(pairs),'completed_records':0,'teacher_screening':'not_performed','v6_curation_complete':False}
        atomic_json(manifest_path,manifest)
        def process(node):
            record_id=node['record_id'];record_path=output/'records'/(record_id+'.json');contact_path=output/'contacts'/(record_id+'.npz')
            if record_path.exists():
                saved=json.loads(record_path.read_text())
                if saved['node']!=node:raise ValueError('Cached record identity changed')
                if saved['status']=='complete':
                    if not contact_path.is_file() or sha256_file(contact_path)!=saved['contact_sha256']:raise ValueError('Cached contact changed')
                    return saved
                if not retry_failed:return saved
            attempts=[]
            for source_id in node['source_ids']:
                match=re.fullmatch(r'([0-9][A-Za-z0-9]{3})_(.+)',source_id)
                if not match:
                    attempts.append({'source_id':source_id,'reason':'unsupported_source_id'});continue
                pdb_id,chain_id=match.groups()
                try:
                    structure,download=cache.fetch(pdb_id)
                    contact,evidence=parse_chain(structure,pdb_id,chain_id,node['sequence'])
                    _save_contact(contact_path,node['sequence'],contact)
                    result={'status':'complete','node':node,'selected_source_id':source_id,'download':download,
                            'evidence':evidence,'contact_sha256':sha256_file(contact_path),'rejected_alternatives':attempts}
                    atomic_json(record_path,result);return result
                except Exception as exc:
                    attempts.append({'source_id':source_id,'reason':f'{type(exc).__name__}: {exc}'})
            result={'status':'rejected','node':node,'reason':'no_exact_observed_chain_with_CA','rejected_alternatives':attempts}
            atomic_json(record_path,result);return result
        results={}
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures={executor.submit(process,node):record_id for record_id,node in sorted(nodes.items())}
                for future in as_completed(futures):
                    record_id=futures[future];results[record_id]=future.result()
                    manifest['completed_records']=len(results)
                    if len(results)%20==0 or len(results)==len(nodes):
                        manifest['verified_records']=sum(r['status']=='complete' for r in results.values())
                        atomic_json(manifest_path,manifest)
                        print(f'structures {len(results)}/{len(nodes)} verified={manifest["verified_records"]}',flush=True)
            valid={key for key,result in results.items() if result['status']=='complete'}
            approved=pairs[pairs.peptide_id.isin(valid)&pairs.target_id.isin(valid)]
            used=set(approved.peptide_id)|set(approved.target_id)
            pd.DataFrame([{'record_id':key,'sequence':nodes[key]['sequence'],'source_id':results[key]['selected_source_id'],'structure_url':results[key]['download']['url'],'structure_sha256':results[key]['download']['sha256']} for key in sorted(used)],columns=['record_id','sequence','source_id','structure_url','structure_sha256']).to_csv(output/'teacher_sequences.csv',index=False)
            approved[['target_id','peptide_id','pair_id']].rename(columns={'target_id':'receptor','peptide_id':'peptide'}).to_csv(output/'teacher_pairs.csv',index=False)
            rejections=[]
            for row in pairs[~pairs.index.isin(approved.index)].itertuples():
                rejections.append({'pair_id':row.pair_id,'peptide_id':row.peptide_id,'target_id':row.target_id,
                                   'reason':'unverified_structure','peptide_status':results[row.peptide_id]['status'],'target_status':results[row.target_id]['status']})
            pd.DataFrame(rejections,columns=['pair_id','peptide_id','target_id','reason','peptide_status','target_status']).to_csv(output/'rejections.csv',index=False)
            failures=[{'record_id':key,'source_id':attempt['source_id'],'reason':attempt['reason']} for key,result in sorted(results.items()) for attempt in result['rejected_alternatives']]
            pd.DataFrame(failures,columns=['record_id','source_id','reason']).to_csv(output/'chain_rejections.csv',index=False)
            if sha256_file(source)!=hashes['pairs'] or sha256_file(dictionary)!=hashes['dictionary']:raise RuntimeError('Input changed during structure preparation')
            if sha256_file(__file__) != hashes['implementation'] or sha256_file(parser_path) != hashes['dictionary_parser']:
                raise RuntimeError('Structure preparation implementation changed')
            manifest.update(status='complete',verified_records=len(valid),teacher_records=len(used),verified_pairs=len(approved),rejected_pairs=len(pairs)-len(approved),
                            output_sha256={name:sha256_file(output/name) for name in ['teacher_sequences.csv','teacher_pairs.csv','rejections.csv','chain_rejections.csv']},
                            limitations=['Structure verification only: TPepPro scores and high-confidence-positive exclusion have not been performed.','Observed-coordinate sequences must match exactly; unavailable/mismatched chains remain excluded.','Current RCSB download snapshots are hashed; historical structural versions are not asserted.'])
            atomic_json(manifest_path,manifest);return manifest
        except BaseException as exc:
            manifest.update(status='failed',error=str(exc));atomic_json(manifest_path,manifest);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--resume',action='store_true');parser.add_argument('--retry-failed',action='store_true');parser.add_argument('--limit-pairs',type=int)
    args=parser.parse_args();print(json.dumps(prepare(**vars(args)),indent=2))
if __name__=='__main__':main()
