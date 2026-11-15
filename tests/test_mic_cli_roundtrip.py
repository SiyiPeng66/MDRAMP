"""Actual CLI warm-start, frozen scaler and validation isolation smoke regression."""
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
from src.features.scaler import StateFeatureScaler
from src.loop.state_manifest import StateManifest,write_manifest


def test_legacy_unbound_low_dimensional_cli_input_is_rejected(tmp_path):
    ids=['train','valid']
    pd.DataFrame({'record_id':ids,'global_index':[0,1]}).to_csv(tmp_path/'metadata.csv',index=False)
    np.savez(tmp_path/'emb.npz',global_embeddings=np.zeros((2,8),np.float32))
    pd.DataFrame({'private_record_id':ids,'pc_x':[0.,0.]}).to_csv(tmp_path/'features.csv',index=False)
    pd.DataFrame({'peptide_id':ids,'mu_prior':[.5,.5],'mu_mem':[.5,.5],'mu_int':[.5,.5]}).to_csv(tmp_path/'means.csv',index=False)
    StateFeatureScaler.fit(np.zeros((2,1)),['pc_x']).save(tmp_path/'scaler.json')
    pd.DataFrame({'peptide_id':ids,'mic_log2':[4.,8.],'mic_censored':[False,True], 'round':['INIT','INIT'],'split':['train','validation'],'released_at':['2020-01-01T00:00:00Z']*2,'qc_pass':[True,True]}).to_csv(tmp_path/'ledger.csv',index=False)
    common=['--mic-ledger','ledger.csv','--peptide-features','features.csv','--peptide-embeddings','emb.npz','--peptide-embedding-metadata','metadata.csv','--expert-means','means.csv','--scaler','scaler.json']
    common=[str(tmp_path/v) if v.endswith(('.csv','.npz','.json')) else v for v in common]
    write_manifest(StateManifest(state_name='M_R0',parent_state=None,visible_round='R0',data_cutoff='2021-01-01T00:00:00Z'),tmp_path/'manifest0.json')
    cmd=[sys.executable,'-m','src.train.train_mic_head',*common,'--state-manifest',str(tmp_path/'manifest0.json'),'--current-round','R0','--output',str(tmp_path/'head0.pt'),'--epochs','1']
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'frozen_scorer_artifacts missing esm_checkpoint' in result.stderr
    assert not (tmp_path/'head0.pt').exists()
    # The strict 1536+32 two-state warmstart positive path is exercised in
    # test_paper_mic_scorer_contract.py with complete measured provenance.
