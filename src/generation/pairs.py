"""Construct reconstruction pairs within existing public homology splits."""
import argparse
import random
from pathlib import Path
import pandas as pd
from src.generation.policy import encode, AA
from src.generation.runtime import validate_training_frame,destinations,atomic_text
from src.utils.hash import sha256_file
import json
from src.optimization.pep_b_genetic import edit_distance

def make_pairs(frame, seed, edits=1):
    validate_training_frame(frame)
    if not {'sequence','split','cluster_id'}.issubset(frame): raise ValueError('sequence, split, cluster_id required')
    if (frame.groupby('cluster_id').split.nunique()>1).any(): raise ValueError('Homology split leakage')
    if edits<1 or edits>3: raise ValueError('Choose 1–3 perturbations')
    rng=random.Random(seed); rows=[]
    ownership=dict(zip(frame.sequence,frame.split))
    for row in frame[frame.split.isin(['train','validation'])].to_dict('records'):
        target=row['sequence']; encode(target)
        for attempt in range(100):
            chars=list(target)
            for _ in range(edits):
                operation=rng.choice(['substitution','insertion','deletion']); i=rng.randrange(len(chars))
                if operation=='substitution': chars[i]=rng.choice(AA.replace(chars[i],''))
                elif operation=='insertion' and len(chars)<50: chars.insert(i,rng.choice(AA))
                elif operation=='deletion' and len(chars)>10: chars.pop(i)
            parent=''.join(chars)
            if 1<=edit_distance(parent,target)<=min(int(.3*len(parent)),6) and (parent not in ownership or ownership[parent]==row['split']): break
        else: raise ValueError('Unable to construct eligible perturbation')
        ownership[parent]=row['split']
        rows.append({**row,'parent':parent,'pair_kind':'synthetic_reconstruction'})
    result=pd.DataFrame(rows)
    validate_training_frame(result,True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--edits',type=int,required=True,help='Engineering choice; manuscript does not prescribe perturbation count')
    a=p.parse_args(); input_hash=sha256_file(a.input)
    with destinations([a.output,a.output+'.manifest.json'],inputs=[a.input]):
        result=make_pairs(pd.read_csv(a.input),a.seed,a.edits)
        if sha256_file(a.input)!=input_hash:raise ValueError('Pair construction input changed')
        atomic_text(a.output,result.to_csv(index=False))
        atomic_text(a.output+'.manifest.json',json.dumps({'arguments':vars(a),'input_sha256':input_hash,'output_sha256':sha256_file(a.output),'pair_count':len(result),'purpose':'synthetic_reconstruction_not_MIC_or_improvement_labels','split_counts':result.split.value_counts().to_dict()},indent=2))
if __name__=='__main__': main()
