import torch
from types import SimpleNamespace
from mdramp.prospective_experts import score_experts


def test_shared_expert_formula_topfive_and_sample_std():
    class Expert:
        def __call__(self,*args):return {'mean':torch.tensor([.6]),'std':torch.tensor([.1])}
    class Pair:
        def __init__(self,k):self.k=k
        def pair_logits(self,p,t,c):return t[:,0]+self.k*.1
    logits=torch.arange(7,dtype=torch.float32)/3
    interaction=SimpleNamespace(members=[Pair(k) for k in range(5)])
    args=(torch.ones(1,2),torch.ones(1,1),Expert(),Expert(),interaction,logits[:,None],[str(i) for i in range(7)])
    actual=score_experts(*args,pair_batch_size=2);other=score_experts(*args,pair_batch_size=10)
    values=[]
    for k in range(5):
        z=logits[-5:]+k*.1;q=z.sigmoid();values.append(.8*(z.softmax(0)*q).sum()+.2*(1-(1-q).prod()))
    values=torch.stack(values)
    torch.testing.assert_close(actual['mu_int'][0],values.mean());torch.testing.assert_close(actual['sigma_int'][0],values.std(unbiased=True))
    torch.testing.assert_close(actual['mu_int'],other['mu_int']);assert actual['top_target_id']==['6']


def test_features_cli_does_not_shadow_main_config(monkeypatch,capsys):
    import pytest
    from mdramp.prospective_features import main
    monkeypatch.setattr('sys.argv',['prospective_features','--help'])
    with pytest.raises(SystemExit) as stopped:main()
    assert stopped.value.code==0
    help_text=capsys.readouterr().out
    assert '--scorer-config SCORER_CONFIG' in help_text
    assert '--config CONFIG' not in help_text


def test_peptide_embedding_identity_and_indices_are_strict():
    import numpy as np
    import pandas as pd
    import pytest
    from src.features.embedding_store import EmbeddingStore
    from mdramp.prospective_features import aligned_embeddings
    frame=pd.DataFrame({'peptide_id':['a','b'],'sequence':['AAAA','KKKK']})
    metadata=pd.DataFrame({'record_id':['b','a'],'sequence':['KKKK','AAAA'],'global_index':[1,0]})
    store=EmbeddingStore(metadata,np.zeros((2,1536),np.float32),None)
    assert aligned_embeddings(store,frame,'peptide_id').shape==(2,1536)
    for indexes in ([-1,0],[0,0],[.5,1],[0,2]):
        store.metadata=metadata.assign(global_index=indexes)
        with pytest.raises(ValueError,match='index'):aligned_embeddings(store,frame,'peptide_id')
    store.metadata=metadata.assign(sequence=['AAAA','KKKK'])
    with pytest.raises(ValueError,match='identity'):aligned_embeddings(store,frame,'peptide_id')


def test_embedding_provenance_checks_checkpoint_and_output_hashes(tmp_path):
    import json
    import pytest
    from src.utils.hash import sha256_file
    from mdramp.prospective_features import validate_embedding_provenance
    emb=tmp_path/'x.npz';meta=tmp_path/'x.csv';manifest=tmp_path/'x.json'
    emb.write_bytes(b'fixture');meta.write_text('fixture')
    document={'status':'complete','encoder':'ESM3','embedding_dimension':1536,'binding':{'checkpoint_sha256':'correct'},'output_sha256':{p.name:sha256_file(p) for p in (emb,meta)}}
    manifest.write_text(json.dumps(document));validate_embedding_provenance(manifest,emb,meta,'correct')
    with pytest.raises(ValueError,match='checkpoint'):validate_embedding_provenance(manifest,emb,meta,'wrong')
    meta.write_text('changed')
    with pytest.raises(ValueError,match='artifact'):validate_embedding_provenance(manifest,emb,meta,'correct')
