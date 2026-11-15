"""Batched autoregressive sampling with the standard proposal guardrails."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.generation.cli import load
from src.generation.runtime import eligibility_context,atomic_text
from src.selection.filters import passes_guardrails
from src.utils.hash import sha256_file
from mdramp.panel_probes import make_identity


def draw(model,count,device,batch_size,top_p,temperature):
    from src.generation.policy import BOS,EOS,PAD
    output=[];model.eval()
    with torch.inference_mode():
        for offset in range(0,count,batch_size):
            n=min(batch_size,count-offset);tokens=[[BOS] for _ in range(n)];finished=[False]*n;residues=[[] for _ in range(n)]
            for length in range(51):
                width=max(map(len,tokens));matrix=torch.full((n,width),PAD,dtype=torch.long,device=device)
                for i,row in enumerate(tokens):matrix[i,:len(row)]=torch.tensor(row,device=device)
                logits=model(matrix);last=torch.tensor([len(row)-1 for row in tokens],device=device);next_logits=logits[torch.arange(n,device=device),last]/temperature
                allowed=torch.zeros_like(next_logits,dtype=torch.bool);allowed[:,:20]=length<50;allowed[:,EOS]=length>=10
                next_logits=next_logits.masked_fill(~allowed,-torch.inf)
                sorted_logits,indices=next_logits.sort(descending=True);prob=sorted_logits.softmax(-1);remove=prob.cumsum(-1)-prob>=top_p
                filtered=next_logits.scatter(1,indices,sorted_logits.masked_fill(remove,-torch.inf))
                sampled=torch.distributions.Categorical(logits=filtered).sample().tolist()
                for i,token in enumerate(sampled):
                    if finished[i]:tokens[i].append(PAD);continue
                    tokens[i].append(token)
                    if token==EOS:finished[i]=True
                    else:residues[i].append('ACDEFGHIKLMNPQRSTVWY'[token])
                if all(finished):break
            output.extend(''.join(row) for row in residues)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['checkpoint','reference-sequences','protected-sequences','output','eligible-output']:p.add_argument('--'+k,required=True)
    p.add_argument('--count',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=2);p.add_argument('--device',default='cpu');p.add_argument('--top-p',type=float,default=.95);p.add_argument('--temperature',type=float,default=1.)
    a=p.parse_args()
    a.parent=None;a.parent_family=None
    if min(a.count,a.batch_size,a.threads)<1 or not 0<a.top_p<=1 or a.temperature<=0:raise ValueError('Invalid sampling arguments')
    torch.manual_seed(a.seed);torch.set_num_threads(a.threads);model,checkpoint=load(a.checkpoint,a.device)
    if checkpoint.get('conditional') or checkpoint.get('metadata',{}).get('stage')!='reinforce':raise ValueError('A completed unconditional prospective RL checkpoint is required')
    references=eligibility_context(a,required=True);by_length={}
    for seq in references:by_length.setdefault(len(seq),[]).append(seq)
    source_paths=[a.checkpoint,a.reference_sequences,a.protected_sequences,__file__,Path(__file__).resolve().parents[1]/'src/generation/policy.py',Path(__file__).resolve().parents[1]/'src/selection/filters.py',Path(__file__).resolve().parents[1]/'mdramp/panel_probes.py']
    hashes={str(Path(x).resolve()):sha256_file(x) for x in source_paths};identity=make_identity()
    sequences=draw(model,a.count,a.device,a.batch_size,a.top_p,a.temperature);seen=set();rows=[]
    for i,seq in enumerate(sequences):
        reason='';eligible,reason=passes_guardrails(seq)
        if eligible:
            for length in range(max(0,int(len(seq)*.9)),int(len(seq)/.9)+1):
                if any(identity(seq,ref)>=.9 for ref in by_length.get(length,[])):
                    eligible=False;reason='homology_or_previous_testing';break
        duplicate=seq in seen;seen.add(seq)
        if duplicate:eligible=False;reason='duplicate_proposal'
        rows.append({'candidate_id':f'proposal_{i:05}','sequence':seq,'eligible':eligible,'eligibility_reason':reason,'duplicate_proposal':duplicate,'candidate_pool_member':eligible})
    frame=pd.DataFrame(rows);out=Path(a.output).absolute();eligible_path=Path(a.eligible_output).absolute()
    if out.exists() or eligible_path.exists():raise ValueError('New output paths required')
    for path in [out,eligible_path]:path.parent.mkdir(parents=True,exist_ok=True)
    atomic_text(str(out),frame.to_csv(index=False));atomic_text(str(eligible_path),frame[frame.candidate_pool_member].to_csv(index=False))
    manifest={'status':'complete','arguments':vars(a),'model_stage':checkpoint['metadata']['stage'],'reference_count':len(references),'proposals':len(frame),'eligible_unique':int(frame.candidate_pool_member.sum()),'identity_method':'Bio.Align global identity at >=90%, validated equivalent to runtime global DP decision on peptide cases','input_sha256':hashes,'output_sha256':{str(x):sha256_file(x) for x in [out,eligible_path]}}
    atomic_text(str(out)+'.manifest.json',json.dumps(manifest,indent=2));print(json.dumps({'proposals':len(frame),'eligible_unique':manifest['eligible_unique']}))
if __name__=='__main__':main()
