"""Wait for external training completion, then bind immutable released artifacts.

The input is a static observer specification, not the changing training states.
Subsequent workflow stages declare the released files as ordinary immutable inputs.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time
import torch
import pandas as pd
from mdramp.calibration_train import fresh_output,sha,_json
from src.generation.runtime import reference_identity


def inspect(spec,base):
    evidence={};details=[]
    for item in spec['generators']:
        output=(base/item['checkpoint']).resolve();state_path=(base/item['training_state']).resolve()
        if not output.is_file() or not state_path.is_file():return None
        lock_path=output.with_name(output.name+'.lock')
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return None
            digest=sha(state_path);state=torch.load(state_path,map_location='cpu',weights_only=True)
            destinations=state['binding'].get('destinations',{})
            if destinations.get('output')!=str(output) or destinations.get('state')!=str(state_path) or state['binding'].get('conditional')!=item['conditional']:raise ValueError('Observer/state run identity mismatch')
            if not isinstance(item['max_epochs'],int) or item['max_epochs']<1:raise ValueError('Invalid SFT epoch budget')
            if state['next_epoch']<item['max_epochs'] and state['stale']<state['binding']['patience']:return None
            if sha(output)!=state['best_sha256']:raise ValueError('Completed SFT reference/state hash mismatch')
            checkpoint=torch.load(output,map_location='cpu',weights_only=True)
            identity=reference_identity(checkpoint)
            if checkpoint['metadata']['arguments'].get('epochs')!=item['max_epochs']:raise ValueError('Observer budget differs from actual SFT budget')
            if state['binding'].get('data_sha256')!=identity['data_sha256']:raise ValueError('Reference data differs from training state')
            if checkpoint['conditional']!=item['conditional']:raise ValueError('SFT task differs from observer specification')
            if sha(state_path)!=digest: return None
            evidence[str(output)]=sha(output);evidence[str(state_path)]=digest
            details.append({'checkpoint':str(output),'next_epoch':state['next_epoch'],'stale':state['stale'],'reference_identity':identity})
    for name in spec['artifacts']:
        path=(base/name).absolute()
        if not path.is_file():return None
        evidence[str(path)]=sha(path)
    marker=(base/spec['expert_completion_marker']).absolute()
    if not marker.is_file():return None
    if json.loads(marker.read_text()).get('status')!='paper_expert_ensembles_complete':return None
    evidence[str(marker)]=sha(marker)
    for name in spec.get('expert_checkpoints',[]):
        path=(base/name).resolve()
        if not path.is_file():return None
        evidence[str(path)]=sha(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        meta=checkpoint['metadata'];config=meta['training_config']
        if config.get('epochs')!=200 or len(set(meta.get('seeds',[])))!=5:raise ValueError('Incomplete expert training metadata')
        expected_batch=32 if meta.get('architecture')=='ProjectedPairAffinityExpert' else 64
        if config.get('batch_size')!=expected_batch:raise ValueError('Wrong expert batch provenance')
    if any(sha(path)!=digest for path,digest in evidence.items()):return None
    return {'status':'completed_external_training_verified','artifact_sha256':evidence,'generators':details}


def run(config,output_dir,timeout=86400,poll_seconds=10):
    path=Path(config).resolve();digest=sha(path);spec=json.loads(path.read_text());base=(path.parent/spec.get('workspace','.')).resolve()
    if timeout<0 or not 0<poll_seconds<=60:raise ValueError('Invalid wait settings')
    deadline=time.monotonic()+timeout
    with fresh_output(output_dir) as output:
        while True:
            if sha(path)!=digest:raise ValueError('Observer specification changed')
            result=inspect(spec,base)
            if result is not None:
                frames=[]
                for entry in spec.get('reference_sources',[]):
                    source=(base/entry).absolute();source_hash=sha(source);frame=pd.read_csv(source)
                    if 'sequence' not in frame or frame.sequence.isna().any():raise ValueError('Invalid reference source')
                    if 'split' not in frame:frame['split']='previously_tested'
                    frame['reference_source']=str(source);frames.append(frame[['sequence','split','reference_source']])
                    if sha(source)!=source_hash:raise ValueError('Reference source changed')
                    result['artifact_sha256'][str(source)]=source_hash
                if frames:
                    pd.concat(frames,ignore_index=True).drop_duplicates().to_csv(output/'references.csv',index=False)
                    result['references_sha256']=sha(output/'references.csv')
                result.update(specification_sha256=digest,implementation_sha256=sha(__file__))
                _json(output/'receipt.json',result);return result
            if time.monotonic()>=deadline:raise TimeoutError('External training has not completed')
            print('Waiting for completed, unlocked training artifacts',flush=True)
            time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--timeout',type=float,default=86400);parser.add_argument('--poll-seconds',type=float,default=10)
    args=parser.parse_args();run(args.config,args.output_dir,args.timeout,args.poll_seconds)

if __name__=='__main__':main()
