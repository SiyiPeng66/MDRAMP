"""Portable model roles and explicit atomic FinalModel promotion."""
import json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path(os.environ.get('MDRAMP_ASSETS',ROOT/'assets')).resolve()
KEYS=('esm_checkpoint','scaler','amp_prior','membrane','target_affinity','target_embeddings','target_embedding_metadata','panel_manifest','mic_checkpoint')
def resolve_model(name='FinalModel'):
 from mdramp.download_assets import ensure_assets
 registry=ASSETS/'models.json'
 if name in ('BaseModel','FinalModel'):
  ensure_assets(ASSETS,[registry])
  doc=json.loads(registry.read_text());path=(ASSETS/doc[name]).absolute();kind=name
  ensure_assets(ASSETS,[path])
 else:
  requested=Path(name).expanduser().resolve();path=requested if requested.suffix=='.json' else requested.parent/'scorer.json';kind='Custom'
  if requested.suffix=='.pt' and path.exists():
   c=json.loads(path.read_text())
   if (path.parent/c.get('mic_checkpoint','')).resolve()!=requested:raise ValueError('Custom checkpoint does not match companion scorer.json')
 if not path.is_file():raise ValueError(f'Model bundle not found: {path}')
 c=json.loads(path.read_text());resolved={k:str((path.parent/c[k]).absolute()) for k in KEYS if k in c}
 generator=(path.parent/c.get('generator','generator.pt')).absolute()
 ensure_assets(ASSETS,[*resolved.values(),generator])
 for k,v in resolved.items():
  if not Path(v).is_file():raise ValueError(f'Missing model asset {k}: {v}')
 return {'scorer_config':str(path),'checkpoint':resolved.get('mic_checkpoint'),'kind':kind,'generator':str((path.parent/c.get('generator','generator.pt')).absolute()),'config':{**c,**resolved}}
def promote_final(scorer_config):
 path=Path(scorer_config).resolve();bundle=resolve_model(str(path))
 if not bundle['checkpoint']:raise ValueError('FinalModel requires a calibrated MIC head')
 from mdramp.preference_scorer import PreferenceSequenceScorer
 PreferenceSequenceScorer(str(path))
 registry=ASSETS/'models.json';doc=json.loads(registry.read_text());doc['FinalModel']=os.path.relpath(path,ASSETS)
 fd,temp=tempfile.mkstemp(dir=ASSETS,prefix='.models-')
 try:
  with os.fdopen(fd,'w') as f:json.dump(doc,f,indent=2)
  os.replace(temp,registry)
 finally:
  if os.path.exists(temp):os.unlink(temp)
