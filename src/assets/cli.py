"""Register, verify and fetch immutable model assets without implicit downloads."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import os
import tempfile
import fcntl
from pathlib import Path

ALIASES = {'esm3': 'biohub/esm3-sm-open-v1', 'protbert': 'Rostlab/prot_bert_bfd'}


def inventory(path):
    path = Path(path).expanduser().resolve()
    if not path.exists(): raise ValueError(f'Missing asset: {path}')
    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() and '.git' not in p.parts and '.cache' not in p.parts)
    if not files: raise ValueError(f'Empty asset directory: {path}')
    result = []
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''): digest.update(block)
        result.append({'path': file.name if path.is_file() else str(file.relative_to(path)), 'bytes':file.stat().st_size,'sha256':digest.hexdigest()})
    return result


def validate_registry(payload):
    if not isinstance(payload,dict) or payload.get('schema_version') != 1 or not isinstance(payload.get('assets'),dict):
        raise ValueError('Unsupported or malformed asset registry schema')
    for name,item in payload['assets'].items():
        if not isinstance(name,str) or not name or not isinstance(item,dict): raise ValueError('Invalid asset entry')
        if not all(isinstance(item.get(k),str) and item[k] for k in ('path','source','revision')): raise ValueError('Missing asset provenance')
        if not isinstance(item.get('files'),list) or not item['files']: raise ValueError('Missing asset inventory')
        paths=set()
        for record in item['files']:
            if not isinstance(record,dict): raise ValueError('Invalid inventory record')
            path=record.get('path')
            if not isinstance(path,str) or not path or Path(path).is_absolute() or '..' in Path(path).parts or path in paths: raise ValueError('Invalid inventory path')
            paths.add(path)
            if type(record.get('bytes')) is not int or record['bytes']<0 or not re.fullmatch(r'[0-9a-f]{64}',str(record.get('sha256',''))): raise ValueError('Invalid inventory hash or size')
    return payload


def read_registry(path):
    path=Path(path)
    return validate_registry(json.loads(path.read_text())) if path.exists() else {'schema_version':1,'assets':{}}


def register(registry, name, path, source, revision):
    path=Path(path).expanduser().resolve()
    if not all(isinstance(v,str) and v.strip() for v in (name,source,revision)): raise ValueError('Name, source and revision are required')
    entry={'path':str(path),'source':source,'revision':revision,'files':inventory(path)}
    registry=Path(registry);registry.parent.mkdir(parents=True,exist_ok=True)
    with registry.with_suffix(registry.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        payload=read_registry(registry)
        if name in payload['assets']: raise ValueError(f'Asset {name!r} already registered; use a new versioned name')
        payload['assets'][name]=entry
        validate_registry(payload)
        fd, temporary=tempfile.mkstemp(prefix=registry.name+'.',dir=registry.parent)
        try:
            with os.fdopen(fd,'w') as handle:
                json.dump(payload,handle,indent=2,sort_keys=True);handle.flush();os.fsync(handle.fileno())
            os.replace(temporary,registry)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    return entry


def status(registry, name=None):
    payload=read_registry(registry)
    if name and name not in payload['assets']: raise ValueError(f'Unknown asset: {name}')
    results={}
    for key,item in payload['assets'].items():
        if name and name != key: continue
        try:
            actual=inventory(item['path'])
            results[key]={'ok':actual==item['files'],'path':item['path'],'revision':item['revision'],'bytes':sum(f['bytes'] for f in actual)}
        except (ValueError,OSError) as error: results[key]={'ok':False,'error':str(error)}
    return results


def doctor():
    packages={}
    for name in ['torch','numpy','pandas','esm','transformers','huggingface-hub','scikit-learn','dgl']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':packages,
            'note':'Package presence does not validate external teacher runtime compatibility or GPU capacity.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry',default='outputs/assets/registry.json')
    sub=parser.add_subparsers(dest='command',required=True)
    reg=sub.add_parser('register');reg.add_argument('--name',required=True);reg.add_argument('--path',required=True)
    reg.add_argument('--source',required=True);reg.add_argument('--revision',required=True)
    for command in ['status','check']:
        check=sub.add_parser(command);check.add_argument('--name')
    sub.add_parser('doctor')
    fetch=sub.add_parser('fetch');fetch.add_argument('--alias',choices=ALIASES,required=True)
    fetch.add_argument('--revision',required=True,help='Full 40-character immutable Hugging Face commit SHA')
    fetch.add_argument('--cache-dir',required=True);fetch.add_argument('--name')
    fetch.add_argument('--allow-network',action='store_true',help='Explicitly permit downloading missing assets')
    fetch.add_argument('--include',action='append',help='Optional snapshot allow-pattern, repeatable; recorded in provenance')
    for child in sub.choices.values(): child.add_argument("--registry",default=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.command=='doctor': result=doctor()
    elif args.command=='register': result=register(args.registry,args.name,args.path,args.source,args.revision)
    elif args.command in ('status','check'):
        result=status(args.registry,args.name)
        print(json.dumps(result,indent=2))
        if args.command=='check' and (not result or not all(v['ok'] for v in result.values())): raise SystemExit(1)
        return
    else:
        if not re.fullmatch(r'[0-9a-fA-F]{40}',args.revision): parser.error('--revision must be an immutable 40-character commit SHA')
        from huggingface_hub import snapshot_download
        root=snapshot_download(repo_id=ALIASES[args.alias],revision=args.revision,cache_dir=str(Path(args.cache_dir).expanduser()),local_files_only=not args.allow_network,allow_patterns=args.include)
        result=register(args.registry,args.name or f'{args.alias}-{args.revision[:12]}',root,
            json.dumps({'repo_id':ALIASES[args.alias],'allow_patterns':args.include}),args.revision)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
