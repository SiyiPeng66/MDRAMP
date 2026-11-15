"""Synthetic-only numerical integration of the public production training CLI.

No fixture represents a scientific experimental observation or learned ESM embedding.
"""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.features.physicochemical import compute_descriptors
from src.loop.state_manifest import StateManifest, write_manifest
from src.utils.hash import sha256_file
import json


def test_production_training_to_cached_prediction(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    def run(command,**options):
        argv=[sys.executable,'-m','mdramp',*command.split()]
        for key,value in options.items():argv.extend(['--'+key.replace('_','-'),str(value)])
        result=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=120)
        assert result.returncode==0, f'{command}\n{result.stdout}\n{result.stderr}'
    rng=np.random.default_rng(42)
    ids=[f'p{i}' for i in range(12)]
    sequences=['KALWQKALWQK'+aa for aa in 'ACDEFGHIKLMN']
    def frame(ids,sequences):
        return pd.DataFrame([{'sequence':seq,'peptide_id':pid,'cluster_id':f'cluster-{pid}','split':'train',**compute_descriptors(seq)} for pid,seq in zip(ids,sequences)])
    positive=frame(ids,sequences);positive.to_csv(tmp_path/'positive.csv',index=False)
    background=frame([f'b{i}' for i in range(12)],sequences[::-1]);background.to_csv(tmp_path/'background.csv',index=False)
    def embedding(name,records):
        np.savez(tmp_path/f'{name}.npz',global_embeddings=rng.normal(size=(len(records),1536)).astype(np.float32))
        pd.DataFrame({'record_id':records,'global_index':range(len(records))}).to_csv(tmp_path/f'{name}-metadata.csv',index=False)
    embedding('positive',ids);embedding('background',background.peptide_id.tolist());embedding('targets',[f't{i}' for i in range(5)])
    pd.DataFrame({'record_id':ids,'pore_teacher_score':[.9,.1]*6}).to_csv(tmp_path/'teacher.csv',index=False)
    checkpoint=tmp_path/'experts';scaler=checkpoint/'feature_scaler.json'
    run('train experts',positive_features=tmp_path/'positive.csv',positive_embeddings=tmp_path/'positive.npz',positive_embedding_metadata=tmp_path/'positive-metadata.csv',background_features=tmp_path/'background.csv',background_embeddings=tmp_path/'background.npz',background_embedding_metadata=tmp_path/'background-metadata.csv',id_column='peptide_id',pore_scores=tmp_path/'teacher.csv',checkpoint_dir=checkpoint,scaler_output=scaler,epochs=1,batch_size=12)
    pairs=pd.DataFrame([{'peptide_id':pid,'target_id':f't{i%5}','label_type':'positive' if i%2 else 'background','s_aff_pair':.9 if i%2 else .1} for i,pid in enumerate(ids)])
    pairs.to_csv(tmp_path/'pairs.csv',index=False)
    shared=dict(peptide_features=tmp_path/'positive.csv',peptide_embeddings=tmp_path/'positive.npz',peptide_embedding_metadata=tmp_path/'positive-metadata.csv',target_embeddings=tmp_path/'targets.npz',target_embedding_metadata=tmp_path/'targets-metadata.csv',peptide_id_column='peptide_id',scaler=scaler)
    run('train interaction',data_mode='public-benchmark',**shared,pair_scores=tmp_path/'pairs.csv',output=checkpoint/'interaction.pt',epochs=1,batch_size=12)
    run('score interaction',**shared,checkpoint=checkpoint/'interaction.pt',pair_output=tmp_path/'predicted-pairs.csv',aggregate_output=tmp_path/'aggregate.csv')
    scoring=dict(candidates=tmp_path/'positive.csv',features=tmp_path/'scoring-features.csv',embeddings=tmp_path/'positive.npz',embedding_metadata=tmp_path/'positive-metadata.csv',id_column='peptide_id',scaler=scaler,amp_prior=checkpoint/'amp_prior_ensemble.pt',membrane=checkpoint/'membrane_ensemble.pt',target_affinity=tmp_path/'aggregate.csv')
    run('score experts',**scoring,output=tmp_path/'means.csv')
    means=pd.read_csv(tmp_path/'means.csv')
    assert 'R_t' not in means and 'A_t' not in means
    assert np.isfinite(means[['mu_prior','mu_mem','mu_int']].to_numpy()).all()
    ledger=pd.DataFrame({'peptide_id':ids,'mic_log2':[4.]*11+[8.],'mic_censored':[False]*11+[True],'round':['INIT']*12,'split':['train']*10+['validation']*2,'released_at':['2020-01-01T00:00:00Z']*12,'qc_pass':[True]*12})
    ledger['target_semantics']='measured_mic'
    ledger['target_definition']='panel_best_mic'
    ledger['mic_units']='ug/mL'
    ledger['measurement_type']=['exact']*11+['right_censored']
    ledger.to_csv(tmp_path/'ledger.csv',index=False)
    # Synthetic provenance fixtures exercise binding, not historical authenticity.
    (tmp_path/'esm-test-fixture.bin').write_bytes(b'synthetic engineering fixture; no ESM inference')
    (tmp_path/'panel-manifest.json').write_text(json.dumps({'purpose':'synthetic test fixture only'}))
    frozen={'esm_checkpoint':str(tmp_path/'esm-test-fixture.bin'),'scaler':str(scaler),
      'amp_prior':str(checkpoint/'amp_prior_ensemble.pt'),'membrane':str(checkpoint/'membrane_ensemble.pt'),
      'target_affinity':str(checkpoint/'interaction.pt'),'target_embeddings':str(tmp_path/'targets.npz'),
      'target_embedding_metadata':str(tmp_path/'targets-metadata.csv'),'panel_manifest':str(tmp_path/'panel-manifest.json')}
    inputs={**frozen,'mic_ledger':str(tmp_path/'ledger.csv'),'peptide_features':str(tmp_path/'positive.csv'),
      'peptide_embeddings':str(tmp_path/'positive.npz'),'peptide_embedding_metadata':str(tmp_path/'positive-metadata.csv'),
      'expert_means':str(tmp_path/'means.csv')}
    write_manifest(StateManifest('M_R0',None,'R0','2021-01-01T00:00:00Z',
      input_hashes={k:sha256_file(v) for k,v in inputs.items()},config={'frozen_scorer_artifacts':frozen}),tmp_path/'manifest.json')
    run('train mic',mic_ledger=tmp_path/'ledger.csv',peptide_features=tmp_path/'positive.csv',peptide_embeddings=tmp_path/'positive.npz',peptide_embedding_metadata=tmp_path/'positive-metadata.csv',expert_means=tmp_path/'means.csv',state_manifest=tmp_path/'manifest.json',current_round='R0',scaler=scaler,peptide_id_column='peptide_id',output=checkpoint/'mic.pt',epochs=1,batch_size=10)
    run('score cached',**scoring,mic_checkpoint=checkpoint/'mic.pt',output=tmp_path/'scores.csv')
    scores=pd.read_csv(tmp_path/'scores.csv')
    assert len(scores)==12 and np.isfinite(scores.R_t).all()
    assert scores.R_t.is_monotonic_decreasing
    np.testing.assert_allclose(scores.R_t,scores.p_cons*scores.A_t)
    assert scores.R_t.nunique()>1
    state=torch.load(checkpoint/'mic.pt',weights_only=False)
    assert state['metadata']['paper_prequential_validated']
