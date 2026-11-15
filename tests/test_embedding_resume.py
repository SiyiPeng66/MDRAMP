import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import src.features.build_esm3_embeddings as module
from src.features.esm3_encoder import ESM3Embedding
from src.features.embedding_store import EmbeddingStore


def setup(tmp_path):
    data=tmp_path/'input.csv';pd.DataFrame({'id':['p1','p2'],'sequence':['ACDEFGHIKL','KALWQKALWQ']}).to_csv(data,index=False)
    weights=tmp_path/'weights.pth';weights.write_bytes(b'fixture checkpoint')
    return dict(dataset='test',input_csv=data,output_dir=tmp_path/'output',id_column='id',
                sequence_column='sequence',valid_column=None,model_name='fixture',device='cpu',
                hf_home=tmp_path/'hf',weights_path=str(weights),resume=True)

class FakeEncoder:
    calls=[]
    fail_id=None
    def __init__(self,**kwargs): pass
    def encode_sequence(self,record_id,sequence):
        self.calls.append(record_id)
        if record_id==self.fail_id:raise RuntimeError('fixture interruption')
        array=np.arange(len(sequence)*4,dtype=np.float32).reshape(len(sequence),4)
        return ESM3Embedding(record_id,sequence,array,array.mean(0))


def test_failure_resume_reuses_completed_and_streams_compatible_artifact(tmp_path,monkeypatch):
    args=setup(tmp_path);monkeypatch.setattr(module,'ESM3Encoder',FakeEncoder)
    FakeEncoder.calls=[];FakeEncoder.fail_id='p2'
    with pytest.raises(RuntimeError,match='interruption'):module.build_embeddings(**args)
    assert json.loads((args['output_dir']/'_cache_test/manifest.json').read_text())['completed_records']==1
    FakeEncoder.fail_id=None;FakeEncoder.calls=[]
    module.build_embeddings(**args)
    assert FakeEncoder.calls==['p2']
    store=EmbeddingStore.load(args['output_dir']/'test.npz',args['output_dir']/'test_metadata.csv')
    assert store.global_embeddings.shape==(2,4)
    assert store.residue_by_record_id('p1').shape==(10,4)
    assert np.allclose(store.global_by_record_id(['p1'])[0],store.residue_by_record_id('p1').mean(0))
    FakeEncoder.calls=[];module.build_embeddings(**args);assert not FakeEncoder.calls
    assert json.loads((args['output_dir']/'test_metadata.json').read_text())['status']=='complete'

@pytest.mark.parametrize('changed',['input','model','settings'])
def test_resume_rejects_changed_binding(tmp_path,monkeypatch,changed):
    args=setup(tmp_path);monkeypatch.setattr(module,'ESM3Encoder',FakeEncoder);FakeEncoder.fail_id=None
    module.build_embeddings(**args)
    if changed=='input':args['input_csv'].write_text(args['input_csv'].read_text()+'\n')
    elif changed=='model':Path(args['weights_path']).write_bytes(b'changed')
    else:args['model_name']='other'
    with pytest.raises(ValueError,match='changed'):module.build_embeddings(**args)


def test_changed_entry_rejected_and_checkpoint_required(tmp_path,monkeypatch):
    args=setup(tmp_path);monkeypatch.setattr(module,'ESM3Encoder',FakeEncoder);FakeEncoder.fail_id=None
    module.build_embeddings(**args)
    entry=next((args['output_dir']/'_cache_test').glob('*.npz'));entry.write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):module.build_embeddings(**args)
    args['weights_path']=None
    with pytest.raises(ValueError,match='checkpoint'):module.build_embeddings(**args)


def test_runtime_change_rejects_mixed_embeddings(tmp_path,monkeypatch):
    args=setup(tmp_path);monkeypatch.setattr(module,'ESM3Encoder',FakeEncoder);FakeEncoder.fail_id=None
    module.build_embeddings(**args)
    original=module._runtime_fingerprint
    def changed(path):
        fingerprint=original(path)
        fingerprint['packages']['esm']='changed-version'
        return fingerprint
    monkeypatch.setattr(module,'_runtime_fingerprint',changed)
    with pytest.raises(ValueError,match='changed'):module.build_embeddings(**args)


def test_snapshot_companions_are_bound(tmp_path):
    weight=tmp_path/'snapshot/data/weights/esm3_sm_open_v1.pth'
    weight.parent.mkdir(parents=True);weight.write_bytes(b'fixture')
    config=tmp_path/'snapshot/config.json';config.write_text('{}')
    before=module._runtime_fingerprint(weight)
    assert 'config.json' in before['snapshot_companion_sha256']
    assert not any('weights/' in name for name in before['snapshot_companion_sha256'])
    config.write_text('{"changed": true}')
    assert module._runtime_fingerprint(weight)!=before
    assert before['cache_format_version']==2
    assert set(before['packages']) >= {'esm','torch','numpy','tokenizers'}
