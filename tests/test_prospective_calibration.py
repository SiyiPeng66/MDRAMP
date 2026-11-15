from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import torch
import pytest
from mdramp import calibration_train as core
from mdramp.prospective_calibration import train,SEMANTICS
from mdramp.prospective_scorer import JoinedMIC
from src.features.scaler import StateFeatureScaler


@pytest.mark.parametrize('assigned',[False,True])
def test_adjusted_1571_full_cv_originals_and_scaler(tmp_path,assigned):
    torch.set_num_threads(1)
    seq=pd.DataFrame([dict(peptide_id=f'p{i}',sequence=aa*10,cluster_id=f'c{i}',fold=i) for i,aa in enumerate('ACDEF')])
    obs=seq[['peptide_id','cluster_id','fold']].copy();obs['strain_id']='best_panel';obs['lower_log2']=8.;obs['upper_log2']=float('inf');obs['lower_inclusive']=False;obs['upper_inclusive']=False
    obs.loc[0,['upper_log2','lower_inclusive','upper_inclusive']]=[8.,True,True]
    if assigned:
        obs['target_semantics']='user_assigned_screen_mic';obs['target_divisor']=1.
        obs['source_interval_lower_log2']=obs.lower_log2;obs['source_interval_upper_log2']=obs.upper_log2
    features=seq[['peptide_id','sequence']].copy()
    for i in range(32):features[f'pc_{i}']=float(i)
    columns=[f'pc_{i}' for i in range(32)]
    StateFeatureScaler(np.ones(32),np.ones(32)*2,columns).save(tmp_path/'scaler.json')
    means=seq[['peptide_id','sequence']].copy()
    for name in ['mu_prior','mu_mem','mu_int']:means[name]=.5
    meta=seq[['peptide_id','sequence']].rename(columns={'peptide_id':'record_id'});meta['global_index']=range(5)
    for name,frame in [('sequences',seq),('observations',obs),('features',features),('expert_means',means),('embedding_metadata',meta)]:frame.to_csv(tmp_path/(name+'.csv'),index=False)
    np.savez(tmp_path/'embeddings.npz',global_embeddings=np.ones((5,1536),dtype=np.float32))
    (tmp_path/'expert_manifest.json').write_text(json.dumps({'output_sha256':core.sha(tmp_path/'expert_means.csv'),'frozen_artifact_sha256':{'scaler':core.sha(tmp_path/'scaler.json')},'input_sha256':{str(tmp_path/name):core.sha(tmp_path/name) for name in ['features.csv','embeddings.npz','embedding_metadata.csv']}}))
    args=SimpleNamespace(**{name:str(tmp_path/(name+'.csv')) for name in ['sequences','observations','features','expert_means','embedding_metadata']},embeddings=str(tmp_path/'embeddings.npz'),expert_manifest=str(tmp_path/'expert_manifest.json'),scaler=str(tmp_path/'scaler.json'),output_dir=str(tmp_path/'trained'),state_id='prospective_v1',epochs=1,batch_size=2,hidden_dim=16,seed=11,learning_rate=5e-5,weight_decay=.01,device='cpu',bootstrap_replicates=10)
    train(args)
    rows=pd.read_csv(tmp_path/'trained/oof_predictions.csv')
    assert rows.target_semantics.eq(SEMANTICS).all()
    assert np.allclose(rows.lower_log2,8-np.log2(10))
    assert rows.original_lower_log2.eq(8).all()
    checkpoint=torch.load(tmp_path/'trained/model.pt',weights_only=True)
    from mdramp import prospective_calibration
    assert checkpoint['input_sha256']['implementation']==core.sha(prospective_calibration.__file__)
    assert checkpoint['input_dim']==1571 and checkpoint['mean']==[0.]*1571
    assert checkpoint['state_dict']['layers.2.weight'].shape==(1,16)
    from mdramp.prospective_predict_cached import run as predict,validate_checkpoint
    wrong=dict(checkpoint);wrong['frozen_artifact_sha256']={'scaler':'wrong'}
    with pytest.raises(ValueError,match='bindings differ'):
        validate_checkpoint(wrong,json.loads((tmp_path/'expert_manifest.json').read_text()))
    prediction_args=SimpleNamespace(**vars(args))
    prediction_args.checkpoint=str(tmp_path/'trained/model.pt');prediction_args.output_dir=str(tmp_path/'predicted')
    prediction_args.oof_predictions=str(tmp_path/'trained/oof_predictions.csv');prediction_args.source_rows=None
    predict(prediction_args)
    predicted=pd.read_csv(tmp_path/'predicted/predictions.csv')
    assert len(predicted)==5 and predicted.used_in_final_model_training.all()
    assert predicted.target_divisor.eq(10).all()
    assert predicted.input_target_semantics.eq('user_assigned_screen_mic' if assigned else 'measured_mic').all()
    head=JoinedMIC(checkpoint)
    assert head(torch.ones(2,1536),torch.ones(2,32),torch.ones(2,3)).shape==(2,)
    for fold in range(5):
        saved=torch.load(tmp_path/f'trained/fold_{fold}/esm_pc.pt',weights_only=True)
        assert f'p{fold}' not in saved['train_peptide_ids']


def test_prospective_scorer_semantic_gate_and_snapshot_symlink(tmp_path):
    import pytest
    from mdramp.prospective_scorer import ProspectiveSequenceScorer
    keys=['esm_checkpoint','scaler','amp_prior','membrane','target_affinity','target_embeddings','target_embedding_metadata','panel_manifest']
    config={}
    for key in keys:
        path=tmp_path/key;path.write_text('constructor fixture only');config[key]=str(path)
    snapshot=tmp_path/'snapshot';snapshot.mkdir();link=snapshot/'weights.pth';link.symlink_to(tmp_path/'esm_checkpoint');config['esm_checkpoint']=str(link)
    checkpoint={'task':'prospective_adjusted_best_mic','target_semantics':SEMANTICS,'target_divisor':10,'input_dim':1571,'hidden_dim':16,'activation':'GELU','historical_reproduction':False,'mean':[0.]*1571,'scale':[1.]*1571,'frozen_artifact_sha256':{key:core.sha(path) for key,path in config.items()}}
    config['mic_checkpoint']=str(tmp_path/'mic.pt');torch.save(checkpoint,config['mic_checkpoint'])
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    scorer=ProspectiveSequenceScorer(config_path)
    assert scorer.config['esm_checkpoint']==str(link)
    assert scorer.encoder is None
    checkpoint['target_semantics']='measured_mic';torch.save(checkpoint,config['mic_checkpoint'])
    with pytest.raises(ValueError,match='adjusted prospective'):ProspectiveSequenceScorer(config_path)


def test_scorer_lazy_initialization_preserves_policy_rng(monkeypatch):
    from mdramp.prospective_scorer import ProspectiveSequenceScorer
    scorer=object.__new__(ProspectiveSequenceScorer);scorer.config={'device':'cpu'}
    monkeypatch.setattr(scorer,'_initialize_impl',lambda:torch.rand(100))
    torch.manual_seed(11);before=torch.get_rng_state().clone()
    scorer._initialize()
    assert torch.equal(before,torch.get_rng_state())


def test_assigned_semantics_reject_divided_and_missing_provenance():
    frame=pd.DataFrame({'target_semantics':['user_assigned_screen_mic'],'target_divisor':[10.], 'source_interval_lower_log2':[5.],'source_interval_upper_log2':[8.]})
    with pytest.raises(ValueError,match='original scale'):core.target_metadata(frame)
    frame['target_divisor']=1.
    with pytest.raises(ValueError,match='source intervals'):core.target_metadata(frame.drop(columns='source_interval_lower_log2'))
