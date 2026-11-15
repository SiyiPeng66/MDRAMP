import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from mdramp.cli import config_arguments,main
from mdramp.project import templates
from src.workflow.run import load_config


def test_config_override_and_paths(tmp_path):
    cfg=tmp_path/'train.yaml'
    cfg.write_text(yaml.safe_dump({'schema_version':1,'command':'train mic','options':{'epochs':200,'output':'model.pt','device':'cpu'}}))
    args=config_arguments(str(cfg),'train mic',['epochs=2'])
    assert args[args.index('--epochs')+1]=='2'
    assert args[args.index('--output')+1]==str(tmp_path/'model.pt')
    with pytest.raises(ValueError,match='mismatch'):config_arguments(str(cfg),'train experts',[])
    with pytest.raises(ValueError):config_arguments(str(cfg),'train mic',['bad'])


def test_unified_features_csv_and_cli_precedence(tmp_path):
    source=tmp_path/'peptides.csv';source.write_text('sequence\nGLFDIVKKVVGALGSL\n')
    cfg=tmp_path/'features.yaml'
    cfg.write_text(yaml.safe_dump({'schema_version':1,'command':'features','options':{'input':'peptides.csv','output':'unused.csv'}}))
    entry = Path(__file__).resolve().parents[1] / 'main.py'
    result=subprocess.run([sys.executable,str(entry),'features','--config',str(cfg),'--output','actual.csv'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'actual.csv').exists() and not (tmp_path/'unused.csv').exists()


def test_bootstrap_template_schema(tmp_path):
    for name,text in templates('.').items(): (tmp_path/name).write_text(text)
    config=load_config(tmp_path/'bootstrap.yaml')
    assert len(config['steps'])==6
    assert '--experts-only' in config['steps'][3]['args']


def test_public_unknown_command():
    assert main(['not-a-command'])==2


def test_esm_config_preserves_snapshot_symlink(tmp_path):
    blob = tmp_path / 'blob'; blob.write_bytes(b'weights')
    weights = tmp_path / 'snapshot' / 'data' / 'weights'
    weights.mkdir(parents=True)
    checkpoint = weights / 'esm3_sm_open_v1.pth'
    checkpoint.symlink_to(blob)
    cfg = tmp_path / 'embed.yaml'
    cfg.write_text(yaml.safe_dump({'schema_version': 1, 'command': 'embed',
                                  'options': {'esm-checkpoint': str(checkpoint.relative_to(tmp_path))}}))
    args = config_arguments(str(cfg), 'embed', [])
    assert args == ['--esm-checkpoint', str(checkpoint)]
