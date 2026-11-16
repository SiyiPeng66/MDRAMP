"""Joint shared-weight MIC revision with censored negative constraints."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mdramp.preference_head import model_from_checkpoint,score,rank_error,requested_rank_loss,boolean_column

def main():
 p=argparse.ArgumentParser();p.add_argument('--classification',required=True);p.add_argument('--public-features');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=20000);p.add_argument('--lr',type=float,default=1e-4)
 a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(11)
 root=Path(__file__).resolve().parents[1];f=pd.read_csv(root/'outputs/preference_head_v1/features/design.csv');z=np.load(root/'outputs/preference_head_v1/features/design.npz');x=torch.tensor(z['x']);assert list(z['sequences'])==f.sequence.tolist()
 c=pd.read_csv(a.classification);negative=set(c.loc[c.negative_constraint.astype(str).str.lower()=='true','sequence']);mask=torch.tensor(f.sequence.isin(negative).to_numpy());preserve=~mask; original=torch.tensor(boolean_column(f.is_original))&preserve; additional=preserve&~original
 ck=torch.load(root/'assets/FinalModel/model.pt',weights_only=True,map_location='cpu');m=model_from_checkpoint(ck)
 with torch.no_grad():base=m(x).squeeze().clone()
 groups=boolean_column(f.group_in_600);constraints=[]
 for i,r in f[f.requested_rank.notna()].iterrows():constraints.append((i,torch.tensor(np.flatnonzero((f['round']==r['round']).to_numpy()&groups&(np.arange(len(f))!=i))),int(r.requested_rank)))
 public=None
 if a.public_features:
  pz=np.load(a.public_features); pf=pd.read_csv(root/'outputs/negative_revision/public/train.csv'); ids=pz['sequences'].astype(str).tolist(); assert set(pf.sequence)<=set(ids); public=torch.tensor(pz['x'][[ids.index(t) for t in pf.sequence]],dtype=torch.float32)
 opt=torch.optim.Adam(m.parameters(),lr=a.lr);best=None;hist=[];pc=torch.tensor(f.p_cons.to_numpy(),dtype=torch.float32)
 for epoch in range(a.epochs+1):
  pred=m(x).squeeze();drift=(pred[preserve]-base[preserve]); retention=100*(pred[original]-base[original]).square().mean()+(pred[additional]-base[additional]).square().mean()+10*(pred[original]-base[original]).square().max();neg=torch.relu(8.15-pred[mask]).square().mean();rank=requested_rank_loss(pred,pc,constraints,1e-5)
  loss=retention+100*neg+1000*rank
  if public is not None:loss=loss+100*torch.relu(8.15-m(public).squeeze()).square().mean()
  if epoch%100==0:
   result=score(m,x,f);bad=int((result.predicted_preference_log2_mic[mask.numpy()]<=8).sum());re=rank_error(result,f);pb=0 if public is None else int((m(public).detach().squeeze()<=8).sum());key=(bad+pb,re,float(retention));hist.append(dict(epoch=epoch,negative_failures=bad,public_failures=pb,rank_error=re,retention_mse=float(retention.detach())))
   if best is None or key<best[0]:best=(key,copy.deepcopy(m.state_dict()),epoch)
   if epoch%1000==0:print(hist[-1],flush=True)
  if epoch<a.epochs:opt.zero_grad();loss.backward();opt.step()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);ck.update(state_dict=best[1],state_id='negative_revision_candidate',negative_revision=True,validation_performed=False);torch.save(ck,out/'model.pt');restored=model_from_checkpoint(torch.load(out/'model.pt',weights_only=True));result=score(restored,x,f);result['previous_mic']=np.exp2(base.numpy());result['negative_constraint']=mask.numpy();result.to_csv(out/'scores.csv',index=False);pd.DataFrame(hist).to_csv(out/'training_log.csv',index=False);(out/'summary.json').write_text(json.dumps({'best_key':best[0],'epoch':best[2],'arguments':vars(a),'input_sha256':{str(t):hashlib.sha256(t.read_bytes()).hexdigest() for t in [Path(a.classification),Path(a.public_features),root/'outputs/negative_revision/public/train.csv',root/'outputs/negative_revision/public/split.csv',root/'outputs/negative_revision/public/manifest.json'] if t.exists()},'scope':'public train only; heldout not inspected during selection; pretraining exposed internal MIC holdout'},indent=2))
if __name__=='__main__':main()
