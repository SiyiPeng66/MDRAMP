import json
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from mdramp.calibration_train import (STRAINS, interval_huber, huber_values, load_observations, load_design, metrics, bootstrap, fit_model)


def fixture_inputs(tmp_path):
    seq=pd.DataFrame([dict(peptide_id=f'p{i}',sequence='A'*10+('C'*i),cluster_id=f'c{i}',fold=i) for i in range(5)])
    obs=pd.DataFrame([dict(peptide_id=f'p{i}',strain_id=s,lower_log2=float(i+3),upper_log2=float(i+3),lower_inclusive=True,upper_inclusive=True,status='usable') for i in range(5) for s in STRAINS])
    obs=obs.merge(seq[['peptide_id','cluster_id','fold']],on='peptide_id',validate='many_to_one')
    features=seq[['peptide_id','sequence']].copy()
    for j in range(32):features[f'pc_{j}']=np.arange(5)+j
    meta=seq[['peptide_id','sequence']].rename(columns={'peptide_id':'record_id'});meta['global_index']=np.arange(5)
    seq.to_csv(tmp_path/'sequences.csv',index=False);obs.to_csv(tmp_path/'observations.csv',index=False)
    features.to_csv(tmp_path/'features.csv',index=False);meta.to_csv(tmp_path/'metadata.csv',index=False)
    np.savez(tmp_path/'embeddings.npz',global_embeddings=np.arange(20,dtype=np.float32).reshape(5,4))
    return seq,obs


def test_interval_loss_exact_left_right_and_bounded():
    pred=torch.tensor([2.,3.,1.,4.],requires_grad=True)
    lo=torch.tensor([1.,-float('inf'),2.,1.]);hi=torch.tensor([1.,2.,float('inf'),5.])
    loss=interval_huber(pred,lo,hi)
    assert loss.item()==pytest.approx(.375)
    loss.backward();assert torch.isfinite(pred.grad).all()
    assert np.allclose(huber_values(pred.detach().numpy(),lo.numpy(),hi.numpy()),[.5,.5,.5,0])


def test_input_guards_and_missing_observations(tmp_path):
    seq,obs=fixture_inputs(tmp_path)
    obs.loc[0,['status','lower_log2','upper_log2']]=['missing',np.nan,np.nan]
    obs.to_csv(tmp_path/'observations.csv',index=False)
    _,valid=load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')
    assert len(valid)==4
    seq.loc[1,'cluster_id']='c0';seq.to_csv(tmp_path/'sequences.csv',index=False)
    with pytest.raises(ValueError,match='Cluster split'):load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')


def test_embedding_identity_validation(tmp_path):
    fixture_inputs(tmp_path)
    metadata=pd.read_csv(tmp_path/'metadata.csv');metadata.loc[0,'sequence']='WRONG';metadata.to_csv(tmp_path/'metadata.csv',index=False)
    with pytest.raises(ValueError,match='Embedding sequence mismatch'):load_design(tmp_path/'features.csv',tmp_path/'embeddings.npz',tmp_path/'metadata.csv')
    metadata['global_index']=metadata.global_index.astype(float)
    metadata.loc[0,'global_index']=.5;metadata.to_csv(tmp_path/'metadata.csv',index=False)
    with pytest.raises(ValueError,match='Invalid embedding'):load_design(tmp_path/'features.csv',tmp_path/'embeddings.npz',tmp_path/'metadata.csv')


def test_train_only_scaler_and_threshold_unknowns(tmp_path):
    seq,obs=fixture_inputs(tmp_path)
    seq,obs=load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')
    design=np.arange(10,dtype=np.float32).reshape(5,2);design[4]=10000
    args=SimpleNamespace(device='cpu',hidden_dim=2,learning_rate=.001,weight_decay=.01,epochs=1,batch_size=32)
    _,ckpt=fit_model(design,seq,obs[obs.fold.ne(4)],args,11)
    assert ckpt['mean']==[3.,4.] and 'p4' not in ckpt['train_peptide_ids']
    rows=pd.DataFrame(dict(lower_log2=[5.,4.,-np.inf],upper_log2=[np.inf,6.,5.],lower_inclusive=[False,True,False],pred=[6.,5.,4.]))
    result=metrics(rows,'pred')
    assert result['exact_mae_log2'] is None
    assert result['thresholds']['32']['known_n']==2
    assert result['thresholds']['32']['positive_n']==1


@pytest.mark.parametrize('design_target',[False,True])
def test_full_train_predict_cli_and_no_overwrite(tmp_path,design_target):
    _,observations=fixture_inputs(tmp_path)
    if design_target:
        observations['target_semantics']='design_target_not_measured'
        observations['target_divisor']=5.
        observations['exact_log2']=observations.lower_log2
        for field in ('lower_log2','upper_log2','exact_log2'):
            observations['measured_'+field]=observations[field]
            observations[field]-=np.log2(5.)
        observations.to_csv(tmp_path/'observations.csv',index=False)
    common=['--features',str(tmp_path/'features.csv'),'--embeddings',str(tmp_path/'embeddings.npz'),'--embedding-metadata',str(tmp_path/'metadata.csv')]
    command=[sys.executable,'-m','mdramp.calibration_train','train',*common,'--sequences',str(tmp_path/'sequences.csv'),'--observations',str(tmp_path/'observations.csv'),'--output-dir',str(tmp_path/'trained'),'--epochs','1','--hidden-dim','2','--bootstrap-replicates','10','--threads','1']
    completed=subprocess.run(command,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    oof=pd.read_csv(tmp_path/'trained/oof_predictions.csv')
    assert len(oof)==5 and np.isfinite(oof[['constant','pc','esm_pc']]).all().all()
    for fold in range(5):
        checkpoint=torch.load(tmp_path/f'trained/fold_{fold}/esm_pc.pt',weights_only=False)
        assert f'p{fold}' not in checkpoint['train_peptide_ids']
        assert checkpoint['strain_ids']==['best_panel']
        assert checkpoint['task']=='peptide_best_mic' and checkpoint['output_dim']==1
        assert checkpoint['activation']=='GELU'
        assert not checkpoint['three_expert_means_included'] and not checkpoint['full_manuscript_architecture']
        assert checkpoint['state_dict']['layers.2.weight'].shape[0]==1
    manifest=json.loads((tmp_path/'trained/manifest.json').read_text())
    assert manifest['final_model_prespecified']=='esm_pc' and manifest['paper_prequential_validated'] is False
    assert subprocess.run(command,capture_output=True).returncode!=0
    prediction=subprocess.run([sys.executable,'-m','mdramp.calibration_train','predict',*common,'--checkpoint',str(tmp_path/'trained/final_model.pt'),'--output-dir',str(tmp_path/'predicted')],capture_output=True,text=True)
    assert prediction.returncode==0,prediction.stderr
    result=pd.read_csv(tmp_path/'predicted/predictions.csv')
    assert len(result)==5 and result.peptide_id.is_unique
    assert 'strain_id' not in result
    assert result.task.eq('peptide_best_mic').all()
    assert not result.probability_calibrated.any()
    semantics='design_target_not_measured' if design_target else 'measured_mic'
    assert manifest['target_semantics']==semantics
    assert manifest['target_divisor']==(5 if design_target else 1)
    assert set(result.target_semantics)=={semantics}
    metric_report=json.loads((tmp_path/'trained/metrics.json').read_text())
    assert metric_report['target_semantics']==semantics
    if design_target:
        assert 'predicted_design_target_mic_ug_ml' in result
        assert 'design_target_threshold_32_score' in result
        assert 'predicted_mic_ug_ml' not in result
        assert 'threshold_32_score' not in result
        assert checkpoint['target_semantics']==semantics
    assert bootstrap(oof,10,11)==bootstrap(oof,10,11)
    old_checkpoint=torch.load(tmp_path/'trained/final_model.pt',weights_only=False)
    old_checkpoint.pop('task');old_checkpoint.pop('output_dim')
    torch.save(old_checkpoint,tmp_path/'old.pt')
    rejected=subprocess.run([sys.executable,'-m','mdramp.calibration_train','predict',*common,'--checkpoint',str(tmp_path/'old.pt'),'--output-dir',str(tmp_path/'wrong_prediction')],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'Checkpoint task mismatch' in rejected.stderr
    assert not (tmp_path/'wrong_prediction').exists()


def test_observation_fold_conflict_rejected(tmp_path):
    _,obs=fixture_inputs(tmp_path)
    obs.loc[0,'fold']=1
    obs.to_csv(tmp_path/'observations.csv',index=False)
    with pytest.raises(ValueError,match='contradicts'):
        load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')



def test_design_target_semantic_guards(tmp_path):
    from mdramp.calibration_train import target_metadata
    _,obs=fixture_inputs(tmp_path)
    obs['target_semantics']='design_target_not_measured';obs['target_divisor']=5.
    obs['exact_log2']=obs.lower_log2
    for field in ('lower_log2','upper_log2','exact_log2'):
        obs['measured_'+field]=obs[field]
        obs[field]-=np.log2(5.)
    assert target_metadata(obs)['target_divisor']==5.
    changed=obs.copy();changed.loc[0,'target_semantics']='measured_mic'
    with pytest.raises(ValueError,match='Mixed'):target_metadata(changed)
    changed=obs.copy();changed.loc[0,'target_divisor']=4.
    with pytest.raises(ValueError,match='Mixed'):target_metadata(changed)
    changed=obs.copy();changed.loc[0,'lower_log2']+=1
    with pytest.raises(ValueError,match='do not match'):target_metadata(changed)
    with pytest.raises(ValueError,match='preserved'):target_metadata(obs.drop(columns='measured_lower_log2'))


def test_derived_semantics_cannot_silently_downgrade():
    from mdramp.calibration_train import target_metadata
    frame=pd.DataFrame({'measured_lower_log2':[5.]})
    with pytest.raises(ValueError,match='explicit design'):target_metadata(frame)
    frame['target_semantics']='measured_mic';frame['target_divisor']=1.
    with pytest.raises(ValueError,match='conflict'):target_metadata(frame)


def test_best_panel_duplicate_and_multistrain_rejected(tmp_path):
    _,obs=fixture_inputs(tmp_path)
    duplicate=pd.concat([obs,obs.iloc[:1]],ignore_index=True)
    duplicate.to_csv(tmp_path/'observations.csv',index=False)
    with pytest.raises(ValueError,match='exactly one'):
        load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')
    obs.loc[0,'strain_id']='MRSA';obs.to_csv(tmp_path/'observations.csv',index=False)
    with pytest.raises(ValueError,match='best_panel'):
        load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')


def test_best_panel_producer_to_trainer(tmp_path):
    from mdramp.calibration_best import aggregate, STRAINS as PANEL_STRAINS
    sequences,_=fixture_inputs(tmp_path)
    observations=pd.DataFrame([dict(peptide_id=row.peptide_id,strain_id=strain,status='usable',cluster_id=row.cluster_id,fold=row.fold,lower_log2=float(3+row.fold+j),upper_log2=float(3+row.fold+j),lower_inclusive=True,upper_inclusive=True,provenance='[]') for row in sequences.itertuples() for j,strain in enumerate(PANEL_STRAINS)])
    result=aggregate(sequences,observations)
    result.to_csv(tmp_path/'observations.csv',index=False)
    _,loaded=load_observations(tmp_path/'sequences.csv',tmp_path/'observations.csv')
    assert len(loaded)==5 and loaded.peptide_id.is_unique
    assert loaded.strain_id.eq('best_panel').all()
    assert np.array_equal(loaded.lower_log2,loaded.fold+3)
