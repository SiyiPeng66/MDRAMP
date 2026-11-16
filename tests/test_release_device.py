import pytest
from mdramp.release_cli import parser,main

@pytest.mark.parametrize('command',['train','calibrate','predict','generate'])
def test_device_defaults_and_cuda_choice(command):
    args=[command]+(['--input','sample.csv'] if command in ('predict','calibrate') else [])
    assert parser().parse_args(args).device=='cpu'
    assert parser().parse_args(args+['--device','cuda']).device=='cuda'

def test_cuda_unavailable_fails_before_download(monkeypatch,capsys):
    monkeypatch.setattr('torch.cuda.is_available',lambda:False)
    monkeypatch.setattr('mdramp.release_cli.resolve_model',lambda _:pytest.fail('must not download'))
    assert main(['predict','--input','missing.csv','--device','cuda'])==2
    assert 'CUDA is unavailable' in capsys.readouterr().out
