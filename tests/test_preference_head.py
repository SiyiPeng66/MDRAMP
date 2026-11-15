import numpy as np
import pandas as pd
import pytest
import torch
from mdramp.calibration_train import CalibrationMLP
from mdramp.preference_head import load_design, score, model_from_checkpoint, main


def fixture(tmp_path):
    torch.manual_seed(3)
    model=CalibrationMLP(1571,16).eval()
    x=np.random.default_rng(3).normal(size=(4,1571)).astype('float32')
    with torch.no_grad():base=model(torch.from_numpy(x)).squeeze(-1).numpy()
    seq=['AAA','CCC','DDD','EEE']
    frame=pd.DataFrame({'sequence':seq,'round':['R3','R3','R3',None],'p_cons':[.9,.8,.7,.6],'group_in_600':[True,True,True,False],'baseline_log2':base,'target_log2':base,'requested_rank':[1.,np.nan,np.nan,np.nan]})
    np.savez(tmp_path/'design.npz',x=x,sequences=np.array(seq))
    frame.to_csv(tmp_path/'design.csv',index=False)
    torch.save({'task':'prospective_adjusted_best_mic','input_dim':1571,'hidden_dim':16,'state_dict':model.state_dict()},tmp_path/'base.pt')
    return frame,x,model


def test_inference_ignores_training_targets_and_names(tmp_path):
    f,x,m=fixture(tmp_path)
    before=score(m,torch.from_numpy(x),f)
    f['target_log2']=1000.;f['requested_rank']=99.;f['sequence']=['XXX','YYY','ZZZ','WWW']
    after=score(m,torch.from_numpy(x),f)
    assert before.predicted_preference_log2_mic.equals(after.predicted_preference_log2_mic)
    assert 'target_log2' not in after and 'requested_rank' not in after
    assert pd.isna(after['rank'].iloc[3])


def test_train_disk_reload_and_target_free_prediction(tmp_path):
    f,x,m=fixture(tmp_path)
    common=['--design',str(tmp_path/'design.npz'),'--table',str(tmp_path/'design.csv')]
    main(['train',*common,'--checkpoint',str(tmp_path/'base.pt'),'--output-dir',str(tmp_path/'trained'),'--epochs','2','--check-every','1'])
    ckpt=torch.load(tmp_path/'trained/model.pt',weights_only=True)
    assert ckpt['task']=='preference_mic_head' and not ckpt['inference_sequence_lookup']
    f.drop(columns=['target_log2','requested_rank','baseline_log2']).to_csv(tmp_path/'inference.csv',index=False)
    main(['predict','--design',str(tmp_path/'design.npz'),'--table',str(tmp_path/'inference.csv'),'--checkpoint',str(tmp_path/'trained/model.pt'),'--output-dir',str(tmp_path/'predicted')])
    a=pd.read_csv(tmp_path/'trained/scores.csv');b=pd.read_csv(tmp_path/'predicted/scores.csv')
    assert a.predicted_preference_log2_mic.equals(b.predicted_preference_log2_mic)
    assert a['rank'].equals(b['rank'])


def test_reject_alignment_and_invalid_membership(tmp_path):
    f,x,m=fixture(tmp_path);f=f.iloc[::-1];f.to_csv(tmp_path/'design.csv',index=False)
    with pytest.raises(ValueError,match='order mismatch'):load_design(tmp_path/'design.npz',tmp_path/'design.csv')


def test_design_manifest_binds_baseline_and_features(tmp_path):
    import json
    from mdramp.calibration_train import sha
    from mdramp.preference_head import verify_design_manifest
    fixture(tmp_path)
    checkpoint={'frozen_artifact_sha256':{'panel':'known'}}
    document={'output_sha256':{p.name:sha(p) for p in [tmp_path/'design.npz',tmp_path/'design.csv']},'frozen_artifact_sha256':{'panel':'known'},'input_sha256':{'baseline':sha(tmp_path/'base.pt')},'target_semantics':'explicit_user_preference_targets_not_measured_MIC'}
    (tmp_path/'manifest.json').write_text(json.dumps(document))
    assert verify_design_manifest(tmp_path/'design.npz',tmp_path/'design.csv',checkpoint,sha(tmp_path/'base.pt'))
    with pytest.raises(ValueError,match='baseline checkpoint'):
        verify_design_manifest(tmp_path/'design.npz',tmp_path/'design.csv',checkpoint,'wrong')
    document['frozen_artifact_sha256']['panel']='other'
    (tmp_path/'manifest.json').write_text(json.dumps(document))
    with pytest.raises(ValueError,match='frozen artifacts'):
        verify_design_manifest(tmp_path/'design.npz',tmp_path/'design.csv',checkpoint,sha(tmp_path/'base.pt'))


def test_preference_raw_scorer_strict_semantics(tmp_path,monkeypatch):
    import json
    from mdramp.calibration_train import sha
    from mdramp.preference_scorer import PreferenceSequenceScorer,SEMANTICS
    from mdramp.prospective_scorer import ProspectiveSequenceScorer
    keys=['esm_checkpoint','scaler','amp_prior','membrane','target_affinity','target_embeddings','target_embedding_metadata','panel_manifest']
    config={}
    for k in keys:
        path=tmp_path/k;path.write_text(k);config[k]=str(path)
    model=CalibrationMLP(1571,16)
    checkpoint={'task':'preference_mic_head','target_semantics':SEMANTICS,'target_divisor':None,'input_dim':1571,'hidden_dim':16,'output_dim':1,'activation':'GELU','historical_reproduction':False,'inference_sequence_lookup':False,'inference_rank_override':False,'mean':[0.]*1571,'scale':[1.]*1571,'state_dict':model.state_dict(),'frozen_artifact_sha256':{k:sha(v) for k,v in config.items()},'baseline_checkpoint_sha256':'baseline','training_input_sha256':{'design':'features'},'state_id':'test_preference'}
    torch.save(checkpoint,tmp_path/'head.pt');config['mic_checkpoint']=str(tmp_path/'head.pt');config['scorer_kind']='preference_mic_head'
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    scorer=PreferenceSequenceScorer(path)
    def fake_score(self,sequences):
        for s in sequences:self.details[s]={'predicted_adjusted_log2_mic':2.,'R_t':.5};self.cache[s]=.5
    monkeypatch.setattr(ProspectiveSequenceScorer,'_score',fake_score)
    result=scorer.score_frame(['AAA'])
    assert result.predicted_preference_mic_ug_ml.iloc[0]==4
    assert result.target_semantics.iloc[0]==SEMANTICS
    assert 'predicted_adjusted_log2_mic' not in result
    checkpoint['target_semantics']='user_adjusted_approximate_mic';torch.save(checkpoint,tmp_path/'head.pt')
    with pytest.raises(ValueError,match='Explicit trained preference'):PreferenceSequenceScorer(path)


def test_requested_rank_loss_moves_prediction_into_gap():
    from mdramp.preference_head import requested_rank_loss
    # All equal mechanism scores: increasing MIC moves a peptide down the list.
    pred=torch.tensor([0.,1.,2.],requires_grad=True)
    p=torch.tensor([.9,.9,.9])
    constraints=[(0,torch.tensor([1,2]),2)]
    loss=requested_rank_loss(pred,p,constraints,1e-5);loss.backward()
    assert pred.grad[0]<0  # Gradient descent increases target MIC.
    assert torch.equal(pred.grad[1:],torch.zeros(2))  # Neighbors detached.
    pred=torch.tensor([1.5,1.,2.],requires_grad=True)
    assert requested_rank_loss(pred,p,constraints,1e-5)==0
