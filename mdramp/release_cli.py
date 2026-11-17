"""Minimal user-facing train/calibrate/predict/generate interface."""
import argparse,json,datetime,math,os
from pathlib import Path
import pandas as pd
import torch
from mdramp.release_assets import resolve_model,ROOT
from mdramp.input_formats import read_sequences
from mdramp.calibration_train import fresh_output

def parser():
 p=argparse.ArgumentParser(description='MDRAMP — BaseModel / FinalModel')
 subs=p.add_subparsers(dest='action',required=True)
 for name in ['train','calibrate','predict','generate']:
  q=subs.add_parser(name);q.add_argument('--device',choices=['cpu','cuda'],default='cpu');q.add_argument('--output-dir');q.add_argument('--threads',type=int,default=2);q.add_argument('--seed',type=int,default=11)
  if name!='train':q.add_argument('--model',default='FinalModel')
  if name in ['predict','calibrate']:q.add_argument('--input',required=True)
  if name in ['predict','generate']:q.add_argument('--details',action='store_true')
  if name in ['train','calibrate']:
   q.add_argument('--epochs',type=int,default=100);q.add_argument('--learning-rate',type=float,default=1e-4);q.add_argument('--batch-size',type=int,default=32)
  if name=='calibrate':q.set_defaults(mic_divisor=10.);q.add_argument('--update-final',action='store_true')
  if name=='generate':
   q.add_argument('--exclude',help='Optional FASTA or table of previously tested sequences')
   q.add_argument('--count',type=int,default=100);q.add_argument('--pool-multiplier',type=int,default=3);q.add_argument('--max-proposals',type=int,default=20000);q.add_argument('--temperature',type=float,default=1.);q.add_argument('--top-p',type=float,default=.95)
 return p

def run_predict(a):
 from mdramp.release_runtime import load_runtime
 f=read_sequences(a.input);bundle=resolve_model(a.model)
 if not bundle['checkpoint']:raise ValueError('BaseModel has no quantitative MIC head; calibrate it with experiments first')
 bundle['config']['device']=getattr(a,'device','cpu');runtime=load_runtime(bundle)
 with fresh_output(a.output_dir) as out:
  result=runtime.predict(f);result[['id','sequence','predicted_mic']].to_csv(out/'predictions.csv',index=False)
  if a.details:result.to_csv(out/'details.csv',index=False)
  runtime.verify();write_run(out,a,runtime.hashes)

def write_run(out,a,hashes,**extra):
 (out/'run.json').write_text(json.dumps({'command':a.action,'model':getattr(a,'model',None),'arguments':vars(a),'artifact_sha256':hashes,**extra},indent=2))

def run_generate(a):
 from src.generation.cli import load
 from mdramp.batched_sample import draw
 from src.selection.filters import passes_guardrails
 from mdramp.release_runtime import load_runtime
 bundle=resolve_model(a.model)
 if not bundle['checkpoint']:raise ValueError('Calibrate BaseModel before MIC-ranked generation')
 if a.count<1 or a.pool_multiplier<1 or a.max_proposals<a.count or not 0<a.top_p<=1 or not math.isfinite(a.temperature) or a.temperature<=0:raise ValueError('Invalid generation settings')
 generator=Path(bundle['generator'])
 if not generator.is_file():raise FileNotFoundError(f'Model generator asset missing: {generator}')
 model,ckpt=load(generator,getattr(a,'device','cpu'))
 if ckpt.get('conditional'):raise ValueError('Expected unconditional generator')
 known=set(read_sequences(a.exclude).sequence) if getattr(a,'exclude',None) else set();seen=set();selected=[];attempts=0
 with fresh_output(a.output_dir) as out:
  while len(selected)<a.count*a.pool_multiplier and attempts<a.max_proposals:
   proposals=draw(model,min(64,a.max_proposals-attempts),getattr(a,'device','cpu'),64,a.top_p,a.temperature);attempts+=len(proposals)
   for s in proposals:
    if s not in seen and s not in known and passes_guardrails(s)[0]:selected.append(s)
    seen.add(s)
   if attempts%512==0:print(f'Generated {len(selected)} eligible unique candidates',flush=True)
  if len(selected)<a.count:raise ValueError(f'Only {len(selected)} eligible sequences from {attempts} proposals; increase --max-proposals')
  del model
  if getattr(a,'device','cpu')=='cuda':torch.cuda.empty_cache()
  bundle['config']['device']=getattr(a,'device','cpu');runtime=load_runtime(bundle);chunks=[]
  for start in range(0,len(selected),32):
   seq=selected[start:start+32];f=pd.DataFrame({'id':[f'candidate_{start+i+1:05}' for i in range(len(seq))],'sequence':seq});chunks.append(runtime.predict(f))
  result=pd.concat(chunks).sort_values(['R_t','sequence'],ascending=[False,True]).head(a.count).copy();result.insert(0,'rank',range(1,len(result)+1))
  result[['rank','id','sequence','predicted_mic']].to_csv(out/'candidates.csv',index=False);(out/'candidates.fasta').write_text(''.join(f'>{r.id}\n{r.sequence}\n' for r in result.itertuples()))
  if a.details:result.to_csv(out/'details.csv',index=False)
  runtime.verify();write_run(out,a,runtime.hashes,proposals=attempts,eligible_pool=len(selected),generator=str(generator),method='Autoregressive sampling; physicochemical filtering; exact previous-sequence exclusion; FinalModel reranking. Not a new RL training run or safety certification.')

def main(argv=None):
 a=parser().parse_args(argv)
 if a.threads<1:raise ValueError('threads must be positive')
 if a.device=='cuda':os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
 if a.output_dir is None:a.output_dir=str(ROOT/'runs'/(a.action+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
 try:
  if a.device=='cuda':
   if not torch.cuda.is_available():raise ValueError('CUDA is unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver, or use --device cpu.')
  if a.action=='predict':run_predict(a)
  elif a.action=='generate':run_generate(a)
  elif a.action=='calibrate':
   from mdramp.release_calibration import run
   run(a)
  else:
   from mdramp.release_train import run
   run(a)
 except (ValueError,FileNotFoundError,FileExistsError) as e:print(f'MDRAMP: {e}');return 2
 print(f'Completed: {a.output_dir}');return 0
if __name__=='__main__':raise SystemExit(main())
