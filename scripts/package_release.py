"""Build a self-contained release folder without historical outputs or environments."""
import argparse,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'releases/MDRAMP'));a=p.parse_args();out=Path(a.output).resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 for name in ['main.py','README.md','README-Zh.md','requirements.txt','pyproject.toml','environment.yml','.gitignore']:
  shutil.copy2(ROOT/name,out/name)
 for name in ['mdramp','src','tests','demo','templates']:
  shutil.copytree(ROOT/name,out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.DS_Store'))
 (out/'data').mkdir()
 for name in ['public']:shutil.copytree(ROOT/'data'/name,out/'data'/name,ignore=shutil.ignore_patterns('*.npz','.feature_cache','.hf_cache','.cache'))
 (out/'configs').mkdir()
 for name in ['requirements-runtime.lock.txt']:shutil.copy2(ROOT/'configs'/name,out/'configs'/name)
 # Publish runtime configuration and the panel definition used by tests.
 for name in ['assets/downloads.json','prospective/protein_panel_v2_rules.json']:
  destination=out/'configs'/name;destination.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(ROOT/'configs'/name,destination)
 shutil.copytree(ROOT/'.github',out/'.github')
 (out/'scripts').mkdir();shutil.copy2(__file__,out/'scripts/package_release.py')
 files={str(f.relative_to(out)):{'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in out.rglob('*') if f.is_file()}
 (out/'MANIFEST.json').write_text(json.dumps({'format':1,'files':files},indent=2))
 print(out)
if __name__=='__main__':main()
