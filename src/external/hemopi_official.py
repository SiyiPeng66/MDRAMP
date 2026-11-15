"""Adapter to the official HemoPI2 RF+MERCI executable (model option 2)."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import pandas as pd

def score_batch(sequences, model_root, python):
    if not sequences or any(not sequence or any(c not in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence) for sequence in sequences):
        raise ValueError('Only standard residues supported')
    root=Path(model_root).resolve()
    with tempfile.TemporaryDirectory(prefix='hemopi-official-') as work:
        work=Path(work); source=work/'input.fa'; source.write_text(''.join(f'>candidate_{i}\n{sequence}\n' for i,sequence in enumerate(sequences)))
        env=dict(os.environ); env['PATH']=str(Path(python).absolute().parent)+os.pathsep+env.get('PATH','')
        result=subprocess.run([str(Path(python).absolute()),str(Path(__file__).with_name('hemopi_rf_entry.py')),str(root/'hemopi2_classification.py'),'-i',str(source),'-o','scores.csv','-m','2','-j','1','-d','2','-wd',str(work)],cwd=root,env=env,capture_output=True,text=True)
        if result.returncode: raise RuntimeError(result.stderr+'\n'+result.stdout)
        frame=pd.read_csv(work/'scores.csv')
        if len(frame)!=len(sequences) or 'Hybrid Score' not in frame: raise ValueError('Invalid official HemoPI2 output')
        frame=frame.set_index('SeqID')
        values=[float(frame.loc[f'candidate_{i}','Hybrid Score']) for i in range(len(sequences))]
        if any(not 0<=value<=1 for value in values): raise ValueError('Invalid hybrid score')
        return values

def score(sequence,model_root,python):
    return score_batch([sequence],model_root,python)[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='Official repository directory with model weights')
    p.add_argument('--python',required=True,help='Compatible isolated Python executable')
    p.add_argument('--sequence',required=True)
    a=p.parse_args(); print(score(a.sequence,a.model,a.python))
if __name__=='__main__': main()
