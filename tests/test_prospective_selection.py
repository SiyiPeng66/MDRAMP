"""Independent ranking authority and diversity checks."""
import numpy as np
import pandas as pd
import pytest
from mdramp.prospective_candidates import diverse_indices,merge_fresh_scores


def test_fresh_scores_override_stale_proposal_rank_and_mic():
    proposals=pd.DataFrame({'sequence':['AAA','CCC'],'R_t':[1.,0.],
        'predicted_adjusted_log2_mic':[-20.,20.],'candidate_id':['a','c']})
    scores=pd.DataFrame({'sequence':['CCC','AAA'],'R_t':[.9,.1],
        'predicted_adjusted_log2_mic':[1.,5.]})
    result=merge_fresh_scores(proposals,scores).sort_values('R_t',ascending=False)
    assert result.sequence.tolist()==['CCC','AAA']
    assert result.predicted_adjusted_log2_mic.tolist()==[1.,5.]
    assert result.candidate_id.tolist()==['c','a']


@pytest.mark.parametrize('scores',[
    {'sequence':['AAA'],'R_t':[.5]},
    {'sequence':['AAA','AAA'],'R_t':[.5,.4]},
    {'sequence':['AAA','CCC'],'R_t':[np.nan,.4]},
])
def test_invalid_scoring_identity_cannot_publish(scores):
    with pytest.raises(ValueError):merge_fresh_scores(pd.DataFrame({'sequence':['AAA','CCC']}),pd.DataFrame(scores))


def test_diversity_greedy_preserves_antibacterial_priority():
    vectors=np.array([[1.,0.],[.999,.001],[0.,1.],[-1.,0.]])
    assert diverse_indices(vectors,2,.1)==[0,2]
    with pytest.raises(ValueError):diverse_indices([[0.,0.]],1,.1)


@pytest.mark.parametrize('corruption',[None,'checkpoint','input','array','sequence','incomplete'])
def test_panel_seal_binds_real_encoder_identity(tmp_path,monkeypatch,corruption):
    import json,sys
    from mdramp import prospective_panelseal as seal
    from src.utils.hash import sha256_file as sha
    proteins=tmp_path/'proteins.csv';meta=tmp_path/'metadata.csv';emb=tmp_path/'vectors.npz'
    frame=pd.DataFrame({'record_id':['a','b'],'sequence':['ACDEFGHIKL','KLMNPQRSTV']})
    frame.to_csv(proteins,index=False);frame.assign(global_index=[0,1]).to_csv(meta,index=False)
    np.savez(emb,global_embeddings=np.ones((2,1536),np.float32))
    checkpoint=tmp_path/'weights.pth';checkpoint.write_bytes(b'fixture-weights')
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'status':'frozen_new_prospective_panel','record_ids':['a','b'],'output_sha256':{'proteins.csv':sha(proteins)}}))
    provenance={'status':'complete','encoder':'ESM3','binding':{'checkpoint_sha256':sha(checkpoint),'input_sha256':sha(proteins)},'output_sha256':{p.name:sha(p) for p in (emb,meta)}}
    if corruption=='checkpoint':provenance['binding']['checkpoint_sha256']='wrong'
    if corruption=='input':provenance['binding']['input_sha256']='wrong'
    if corruption=='array':np.savez(emb,global_embeddings=np.zeros((2,1536),np.float32))
    if corruption=='sequence':
        changed=pd.read_csv(meta);changed.loc[0,'sequence']='WRONG';changed.to_csv(meta,index=False)
    if corruption=='incomplete':provenance['status']='running'
    origin=tmp_path/'origin.json';origin.write_text(json.dumps(provenance));output=tmp_path/'sealed.json'
    args={'panel-manifest':panel,'proteins':proteins,'embeddings':emb,'embedding-metadata':meta,'esm-checkpoint':checkpoint,'embedding-provenance':origin,'output':output}
    monkeypatch.setattr(sys,'argv',['seal',*[v for k,p in args.items() for v in ('--'+k,str(p))]])
    if corruption:
        with pytest.raises(ValueError):seal.main()
        assert not output.exists()
    else:
        seal.main();saved=json.loads(output.read_text())
        assert saved['embedding_provenance_sha256']==sha(origin) and saved['historical_reproduction'] is False
        assert saved['artifact_sha256']['target_embeddings']==sha(emb)
        with pytest.raises(FileExistsError):seal.main()
