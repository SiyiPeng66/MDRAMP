"""Independent probe identity, holdout isolation and assembler-contract tests."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from mdramp import panel_probes as probes
from src.data.cluster_split import length_aware_identity


def test_fast_identity_threshold_matches_original_global_dp():
    rng=np.random.default_rng(91);fast=probes.make_identity();aa=np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    pairs=[('A'*9+'C','A'*10),('A'*8+'CC','A'*10),('A'*9,'A'*10),('A'*8,'A'*10),('AC'*25,'CA'*25),('','A')]
    for _ in range(240):
        a=''.join(rng.choice(aa,size=int(rng.integers(10,51))));b=list(a)
        for _ in range(int(rng.integers(0,7))):
            op=int(rng.integers(0,3));i=int(rng.integers(len(b)))
            if op==0:b[i]=str(rng.choice(aa))
            elif op==1:b.insert(i,str(rng.choice(aa)))
            elif len(b)>1:b.pop(i)
        pairs.append((a,''.join(b)))
    for a,b in pairs:
        ref=length_aware_identity(a,b);actual=fast(a,b)
        assert (actual>=.90)==(ref>=.90),(a,b,actual,ref)
        if a and b and min(len(a),len(b))/max(len(a),len(b))>=.9:assert actual==pytest.approx(ref)
        assert fast(a,b)==fast(b,a)


def test_prepare_frozen_holdout_and_private_sequence_only(tmp_path,monkeypatch):
    rng=np.random.default_rng(43);aa=np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    seqs=[''.join(rng.choice(aa,size=20)) for _ in range(80)]
    public=tmp_path/'public.csv';private=tmp_path/'private.csv';generated=tmp_path/'generated.csv';protocol=tmp_path/'protocol.json'
    pd.DataFrame([dict(sequence=s,cluster_id=f'c{i}',split='validation' if i<20 else 'test') for i,s in enumerate(seqs[:40])]).to_csv(public,index=False)
    pd.DataFrame({'sequence':seqs[40:42],'MIC':['DO_NOT_READ','SECRET_LABEL']}).to_csv(private,index=False)
    pd.DataFrame({'sequence':seqs[42:73]+[seqs[42]]}).to_csv(generated,index=False)
    protocol.write_text(json.dumps({'seed':17,'public_cohort_size':4}))
    actual=pd.read_csv;calls=[]
    def read(path,*args,**kwargs):
        if str(path)==str(private):calls.append(kwargs.get('usecols'))
        return actual(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',read)
    for name in ['first','second']:
        probes.prepare(SimpleNamespace(public=public,private_sequences=private,generated=generated,protocol=protocol,output_dir=tmp_path/name))
    assert calls==[['sequence'],['sequence']]
    assert (tmp_path/'first/cohorts.csv').read_bytes()==(tmp_path/'second/cohorts.csv').read_bytes()
    f=actual(tmp_path/'first/cohorts.csv');test=f[f.cohort=='public_test_confirmation'];rest=f[f.cohort!='public_test_confirmation']
    assert len(test)==4 and not set(test.cluster_id)&set(rest.cluster_id)
    assert len(f[f.cohort=='generated_development'])==31
    assert all(length_aware_identity(a,b)<.9 for a in test.sequence for b in rest.sequence)


def test_assembler_rejects_unbound_source_content(tmp_path):
    sequence='ACDEFGHIKLMNPQRSTVWY';p=tmp_path/'probes.csv';pd.DataFrame({'record_id':['probe'],'sequence':[sequence]}).to_csv(p,index=False)
    weights=tmp_path/'weights';weights.write_bytes(b'fixed')
    npz=tmp_path/'vectors.npz';np.savez(npz,global_embeddings=np.zeros((1,1536),np.float32))
    meta=tmp_path/'meta.csv';pd.DataFrame({'record_id':['source'],'sequence':[sequence],'global_index':[0]}).to_csv(meta,index=False)
    provenance=tmp_path/'provenance.json';provenance.write_text(json.dumps({'encoder':'ESM3','status':'complete','embedding_dimension':1536,'binding':{'checkpoint_sha256':probes.sha(weights)},'output_sha256':{npz.name:probes.sha(npz),meta.name:'incorrect'}}))
    cfg=tmp_path/'sources.json';cfg.write_text(json.dumps({'esm_checkpoint':str(weights),'sources':[{'npz':str(npz),'metadata':str(meta),'provenance':str(provenance)}]}))
    with pytest.raises(ValueError,match='source hash mismatch'):
        probes.assemble(SimpleNamespace(probes=p,sources=cfg,output_dir=tmp_path/'out'))
    assert not (tmp_path/'out').exists()
