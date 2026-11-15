"""Development-only comparison of frozen functional panels; no automatic adoption."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mdramp.panel_stability import aggregate,measure,validate_manifest
from mdramp.protein_panel_build import sha,dump
from mdramp.calibration_train import fresh_output


def validate_frozen_design(a,protocol,bindings):
 """Reject confirmation rows and unbound hand-edited scope/cohort files."""
 allowed=['generated_development','public_validation_development']
 if protocol.get('development_cohorts')!=allowed:raise ValueError('Only the two frozen development cohorts are allowed')
 cohort_path=Path(a.cohorts).absolute();manifest_path=cohort_path.parent/'manifest.json'
 cm=json.loads(manifest_path.read_text())
 if cm.get('status')!='frozen_label_free_cohorts' or cm.get('protocol')!=protocol:raise ValueError('Frozen development protocol mismatch')
 if cm.get('output_sha256',{}).get(cohort_path.name)!=sha(cohort_path):raise ValueError('Cohort file is not bound to frozen manifest')
 bindings[str(manifest_path)]=sha(manifest_path)
 for name,digest in cm['output_sha256'].items():
  path=cohort_path.parent/name
  if sha(path)!=digest:raise ValueError('Frozen cohort output changed')
  bindings[str(path)]=digest
 for pathstr in a.panels:
  path=Path(pathstr).absolute();mp=path.parent.parent/'manifest.json';m=json.loads(mp.read_text())
  if m.get('status')!='constructed_pending_independent_review':raise ValueError('Frozen functional-panel construction required')
  key=str(path.relative_to(mp.parent))
  if m['output_sha256'].get(key)!=sha(path):raise ValueError('Panel is not bound to functional selection manifest')
  bindings[str(mp)]=sha(mp)
  for name,digest in m['output_sha256'].items():
   source=mp.parent/name
   if sha(source)!=digest:raise ValueError('Functional selection output changed')
   bindings[str(source)]=digest
  for source,digest in m['input_sha256'].items():
   if sha(source)!=digest:raise ValueError('Functional selection input changed')
   bindings[str(Path(source).absolute())]=digest


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--panels',nargs='+',required=True)
 for key in ['cache-dir','cohorts','protocol','output-dir']:p.add_argument('--'+key,required=True)
 evaluate(p.parse_args())

def evaluate(a):
 torch.set_num_threads(2)
 cache=Path(a.cache_dir).absolute();cm=validate_manifest(cache)
 protocol=json.loads(Path(a.protocol).read_text());cohorts=pd.read_csv(a.cohorts,dtype=str)
 peptides=pd.read_csv(cache/'peptides.csv',dtype=str);targets=pd.read_csv(cache/'targets.csv',dtype=str);fixed=pd.read_csv(cache/'fixed_experts.csv')
 if peptides.record_id.tolist()!=fixed.record_id.astype(str).tolist():raise ValueError('Fixed score alignment mismatch')
 if targets.record_id.duplicated().any() or peptides.record_id.duplicated().any():raise ValueError('Duplicate IDs')
 bindings={str(cache/'manifest.json'):sha(cache/'manifest.json'),str(Path(a.cohorts).absolute()):sha(a.cohorts),str(Path(a.protocol).absolute()):sha(a.protocol),str(Path(__file__).absolute()):sha(__file__)}
 for name,digest in cm['output_sha256'].items():bindings[str(cache/name)]=digest
 for file in ['mdramp/panel_stability.py','src/models/gated_integrator.py']:
  path=Path(__file__).resolve().parents[1]/file;bindings[str(path)]=sha(path)
 validate_frozen_design(a,protocol,bindings)
 groups={};lookup=targets.set_index('record_id');col={x:i for i,x in enumerate(targets.record_id)};row={x:i for i,x in enumerate(peptides.record_id)}
 for path in a.panels:
  path=Path(path).absolute();selected=pd.read_csv(path,dtype=str);name=path.parent.name
  if name in groups or selected.record_id.duplicated().any() or len(selected)<5:raise ValueError('Invalid panel identity')
  if selected.sequence.tolist()!=lookup.loc[selected.record_id].sequence.tolist():raise ValueError('Panel sequence mismatch')
  groups[name]=[col[x] for x in selected.record_id];bindings[str(path)]=sha(path)
 logits=np.load(cache/'logits.npy',mmap_mode='r')
 if logits.shape!=(len(peptides),5,len(targets)):raise ValueError('Logit shape mismatch')
 with fresh_output(a.output_dir) as out:
  dump(out/'frozen_inputs.json',bindings);result={};tables=[]
  for cohort in protocol['development_cohorts']:
   group=cohorts[cohorts.cohort.eq(cohort)].sort_values('record_id')
   if group.record_id.duplicated().any() or len(group)<protocol['top_k']:raise ValueError('Invalid cohort')
   rows=[row[x] for x in group.record_id]
   if group.sequence.tolist()!=peptides.iloc[rows].sequence.tolist() or group.cluster_id.tolist()!=peptides.iloc[rows].cluster_id.tolist():raise ValueError('Cohort identity mismatch')
   base=aggregate(logits[rows],fixed.iloc[rows]);ties=[hashlib.sha256(s.encode()).hexdigest() for s in group.sequence]
   for name,cols in groups.items():
    score=aggregate(logits[rows][:,:,cols],fixed.iloc[rows]);metrics={k:measure(base[k],score[k],group.cluster_id.to_numpy(),ties,protocol,protocol['seed']) for k in ['mu_int','p_cons']}
    result.setdefault(name,{'count':len(cols),'development':{}})['development'][cohort]=metrics
    frame=group.copy();frame['panel']=name
    for k in score:frame[k]=score[k];frame['baseline_'+k]=base[k]
    tables.append(frame)
  for r in result.values():r['passes_all_development_gates']=all(m['passes_all_gates'] for cohort in r['development'].values() for m in cohort.values())
  dump(out/'metrics.json',result);pd.concat(tables,ignore_index=True).to_csv(out/'scores.csv',index=False)
  if any(sha(p)!=v for p,v in bindings.items()):raise ValueError('Inputs changed')
  dump(out/'manifest.json',{'status':'complete','scope':'development_only_no_automatic_adoption_no_confirmation_read','input_sha256':bindings,'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
 print(json.dumps({k:{'count':v['count'],'passes':v['passes_all_development_gates']} for k,v in result.items()}))
if __name__=='__main__':main()
