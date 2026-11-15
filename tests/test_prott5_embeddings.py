"""Explicit synthetic test doubles only; production encoder never fabricates vectors."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from src.features import prott5_embeddings as module


def test_official_tokenization_and_eos_removal():
    captured={}
    class Tokenizer:
        eos_token_id=1
        def batch_encode_plus(self,values,**kwargs):
            captured['values']=values;captured['kwargs']=kwargs
            return {'input_ids':[[5,6,7,8,1]],'attention_mask':[[1]*5]}
    class Model:
        def __call__(self,**kwargs):return SimpleNamespace(last_hidden_state=torch.arange(5*1024,dtype=torch.float32).reshape(1,5,1024))
    encoder=module.ProtT5Encoder.__new__(module.ProtT5Encoder)
    encoder.tokenizer=Tokenizer();encoder.model=Model();encoder.device='cpu'
    result=encoder.encode('AUZB')
    assert captured['values']==['A X X X'] and captured['kwargs']=={'add_special_tokens':True,'padding':'longest'}
    assert result.shape==(4,1024) and result[-1,-1]==4095


def test_cache_resume_and_corruption_are_bound(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir()
    for name in ['config.json','spiece.model','tokenizer_config.json','special_tokens_map.json','pytorch_model.bin']:(model/name).write_bytes(b'test-only artifact')
    data=tmp_path/'input.csv';pd.DataFrame({'id':['a','b','c'],'seq':['ACDE','ACDE','KLMNP']}).to_csv(data,index=False)
    calls=[]
    class FakeEncoder:
        def __init__(self,*args):pass
        def encode(self,seq):calls.append(seq);return np.ones((len(seq),1024),dtype=np.float32)
    monkeypatch.setattr(module,'ProtT5Encoder',FakeEncoder)
    args=['program','--input-csv',str(data),'--id-column','id','--sequence-column','seq','--model-dir',str(model),'--output-dir',str(tmp_path/'output')]
    monkeypatch.setattr(sys,'argv',args);module.main()
    assert calls==['ACDE','KLMNP']
    with np.load(tmp_path/'output/embeddings.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={'a','b','c'} and archive['c'].shape==(5,1024)
    monkeypatch.setattr(sys,'argv',args+['--resume']);module.main();assert len(calls)==2
    cache=next((tmp_path/'output/cache').glob('*.npz'));cache.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):module.main()


def test_invalid_sequence_rejected():
    with pytest.raises(ValueError):module.normalize_sequence('ACD*')
