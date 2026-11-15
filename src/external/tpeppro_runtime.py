"""CPU TPepPro inference using original checkpoint and exact TAGConv operations.

DGL's unweighted TAGConv k=2 is expressed directly in torch; no substitute
features or learned parameters are introduced. Source operator specification:
https://www.dgl.ai/dgl_docs/_modules/dgl/nn/pytorch/conv/tagconv.html
Inputs are original TPepPro 1024-dimensional embeddings and contact maps,
not ESM3 embeddings. This runtime does not fabricate structure inputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.utils.hash import sha256_file

class TAGConv(nn.Module):
    def __init__(self):
        super().__init__(); self.lin=nn.Linear(3072,1024)
    def forward(self,features,contact):
        # DGLGraph(scipy COO) edges are row -> column; edge values ignored.
        adjacency=(contact != 0).to(features.dtype)
        norm=adjacency.sum(0).clamp(min=1).pow(-.5)[:,None]
        stack=[features]
        for _ in range(2): stack.append((adjacency.T @ (stack[-1]*norm))*norm)
        return self.lin(torch.cat(stack,-1))

class ConvsLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv1d(1024,128,3); self.mx1=nn.MaxPool1d(3,3)
        self.conv2=nn.Conv1d(128,128,3); self.mx2=nn.MaxPool1d(3,3)
        self.conv3=nn.Conv1d(128,128,3); self.mx3=nn.MaxPool1d(130,1)
    def forward(self,x):
        x=x.T.unsqueeze(0)
        x=self.mx1(self.conv1(x))
        x=self.mx2(self.conv2(x))
        return self.mx3(self.conv3(x)).squeeze(2)

class TPepProRuntime(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.gcn1=TAGConv(); self.fc_g1=nn.Linear(1024,128)
        self.textcnn=ConvsLayer(); self.textflatten=nn.Linear(128,128)
        self.w1=nn.Parameter(torch.tensor([.5])); self.fc1=nn.Linear(256,512)
        self.BatchNorm1d_1=nn.BatchNorm1d(512); self.fc2=nn.Linear(512,256)
        self.BatchNorm1d_2=nn.BatchNorm1d(256); self.out=nn.Linear(256,1)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        self.eval(); self.requires_grad_(False)
    def encode(self,embedding,contact):
        x=torch.as_tensor(embedding,dtype=torch.float32); c=torch.as_tensor(contact)
        if x.ndim!=2 or x.shape[1]!=1024 or c.shape!=(len(x),len(x)) or len(x)==0:
            raise ValueError('Expected Lx1024 original embeddings and LxL contact map')
        if not torch.isfinite(x).all() or not torch.isfinite(c).all(): raise ValueError('Nonfinite inputs')
        graph=torch.relu(self.fc_g1(torch.relu(self.gcn1(x,c)).amax(0,keepdim=True)))
        padded=torch.zeros(1200,1024); padded[:min(len(x),1200)]=x[:1200]
        text=torch.relu(self.textflatten(self.textcnn(padded)))
        weight=self.w1.sigmoid()
        return (1-weight)*graph+weight*text
    @torch.no_grad()
    def forward(self,receptor_embedding,receptor_contact,peptide_embedding,peptide_contact):
        combined=torch.cat([self.encode(receptor_embedding,receptor_contact),self.encode(peptide_embedding,peptide_contact)],1)
        hidden=torch.relu(self.BatchNorm1d_1(self.fc1(combined)))
        hidden=torch.relu(self.BatchNorm1d_2(self.fc2(hidden)))
        return self.out(hidden).sigmoid().item()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--embeddings',required=True)
    parser.add_argument('--contact-dir',required=True); parser.add_argument('--pairs',required=True,help='CSV receptor,peptide columns')
    parser.add_argument('--output',required=True); parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    model=TPepProRuntime(args.checkpoint); embeddings=np.load(args.embeddings,allow_pickle=False)
    rows=[]; contact_hashes={}
    for pair in pd.read_csv(args.pairs).itertuples(index=False):
        inputs=[]
        for identifier in (pair.receptor,pair.peptide):
            contact_path=Path(args.contact_dir)/(str(identifier)+'.npz')
            contact_hashes[str(identifier)]=sha256_file(contact_path)
            cmap=np.load(contact_path,allow_pickle=False)
            length=len(str(cmap['seq']))
            inputs.extend([embeddings[identifier][:length],cmap['contact']])
        rows.append({'receptor':pair.receptor,'peptide':pair.peptide,'predict_score':model(*inputs)})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output,index=False)
    Path(args.output+'.provenance.json').write_text(json.dumps({'runtime':'original TPepPro weights; torch expression of DGL TAGConv','checkpoint_sha256':sha256_file(args.checkpoint),'embeddings_sha256':sha256_file(args.embeddings),'pairs_sha256':sha256_file(args.pairs),'contact_sha256':contact_hashes,'dgl_execution_parity_verified':False},indent=2))
if __name__=='__main__': main()
