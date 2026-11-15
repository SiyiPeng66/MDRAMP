"""Strict measured/prequential model contract, independent of large ESM loading."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from src.train.train_mic_head import validate_measured_targets, bound_artifacts, FROZEN_ARTIFACTS, train_mic_head
from src.score.frozen_scorer import FrozenSequenceScorer
from src.models.mic_regressor import MICRegressionHead


def ledger():
    return pd.DataFrame(dict(target_semantics=['measured_mic']*2,target_definition=['panel_best_mic']*2,
      mic_units=['ug/mL']*2,measurement_type=['exact','right_censored'],mic_censored=[False,True],mic_log2=[4.,8.]))


def test_targets_reject_design_screen_and_artificial_values():
    assert validate_measured_targets(ledger()).tolist()==[False,True]
    for key,value in [('target_semantics','design_target_not_measured'),('target_divisor',10),
                      ('target_definition','single_strain'),('measurement_type','interval'),('mic_log2',9)]:
        frame=ledger();frame[key]=value
        with pytest.raises(ValueError):validate_measured_targets(frame)
    with pytest.raises(ValueError):validate_measured_targets(ledger().drop(columns='target_semantics'))


def test_generation_constants_cannot_silently_drift():
    scorer=object.__new__(FrozenSequenceScorer)
    for key in ['aggregation_coefficient','kappa','tau','gamma','lambda_syn']:
        scorer.config={key:7}
        with pytest.raises(ValueError,match='fixed'):scorer._validate_constants()
    scorer.config={};scorer._validate_constants()


def test_frozen_input_binding_requires_real_hashes(tmp_path):
    from src.utils.hash import sha256_file
    keys=set(FROZEN_ARTIFACTS)|{'mic_ledger','peptide_features','peptide_embeddings','peptide_embedding_metadata','expert_means'}
    paths={}
    for key in keys:
        path=tmp_path/key;path.write_text(key);paths[key]=str(path)
    manifest=SimpleNamespace(config={'frozen_scorer_artifacts':{k:paths[k] for k in FROZEN_ARTIFACTS}},input_hashes={k:sha256_file(v) for k,v in paths.items()})
    _, hashes=bound_artifacts(manifest,SimpleNamespace(**paths));assert hashes==manifest.input_hashes
    (tmp_path/'expert_means').write_text('changed')
    with pytest.raises(ValueError,match='input_hashes'):bound_artifacts(manifest,SimpleNamespace(**paths))


def test_early_stop_restores_best_and_inputs_stay_frozen():
    torch.manual_seed(11);model=MICRegressionHead(2,1)
    inputs=(np.zeros((2,2),np.float32),np.zeros((2,1),np.float32),np.ones((2,3),np.float32)*.5,np.ones(2,np.float32),np.zeros(2,bool))
    # Zero learning rate keeps validation unchanged, so patience ends after 1 best + 2 stale epochs.
    original={k:v.clone() for k,v in model.state_dict().items()}
    losses=train_mic_head(model,*inputs,epochs=20,batch_size=2,learning_rate=0,validation=inputs,patience=2)
    assert len(losses)==3
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())


def test_legacy_mic_boolean_does_not_establish_compatibility(tmp_path):
    from src.loop.state_manifest import StateManifest,write_manifest
    paths={}
    for key in (*FROZEN_ARTIFACTS,'state_manifest','mic_checkpoint'):
        p=tmp_path/key;p.write_bytes(b'x');paths[key]=str(p)
    write_manifest(StateManifest('M_R0',None,'R0','2020-01-01T00:00:00Z'),paths['state_manifest'])
    torch.save({'metadata':{'paper_prequential_validated':True,'peptide_dim':1536,'physchem_dim':32}},paths['mic_checkpoint'])
    scorer=FrozenSequenceScorer(paths)
    with pytest.raises(ValueError,match='measured panel-best'):scorer.validate_artifacts()
    assert scorer.encoder is None


def test_strict_cli_initialization_and_cumulative_warmstart(tmp_path,monkeypatch):
    """Synthetic engineering fixture only; no claim of historical measurements."""
    from src.train.train_mic_head import main
    from src.features.physicochemical import compute_descriptors
    from src.features.scaler import StateFeatureScaler
    from src.loop.state_manifest import StateManifest,write_manifest
    from src.utils.hash import sha256_file
    paths={k:str(tmp_path/k) for k in set(FROZEN_ARTIFACTS)|{'mic_ledger','peptide_features','peptide_embeddings','peptide_embedding_metadata','expert_means'}}
    for path in paths.values():open(path,'wb').write(b'test-fixture-only')
    columns=list(compute_descriptors('ACDEFGHIKLMNPQRSTVWY'))
    ids=['a','b']; frame=ledger()
    frame['peptide_id']=ids;frame['round']='INIT';frame['split']=['train','validation'];frame['released_at']='2020-01-01T00:00:00Z';frame['qc_pass']=True
    frame.to_csv(paths['mic_ledger'],index=False)
    pd.DataFrame({'private_record_id':ids,**{c:[0.,0.] for c in columns}}).to_csv(paths['peptide_features'],index=False)
    with open(paths['peptide_embeddings'],'wb') as handle:np.savez(handle,global_embeddings=np.zeros((2,1536),np.float32))
    pd.DataFrame({'record_id':ids,'global_index':[0,1]}).to_csv(paths['peptide_embedding_metadata'],index=False)
    pd.DataFrame({'peptide_id':ids,'mu_prior':[.5,.5],'mu_mem':[.5,.5],'mu_int':[.5,.5]}).to_csv(paths['expert_means'],index=False)
    StateFeatureScaler(np.zeros(32),np.ones(32),columns).save(paths['scaler'])
    previous=None
    for round_index in range(2):
        manifest=StateManifest(f'M_R{round_index}',f'M_R{round_index-1}' if round_index else None,f'R{round_index}','2021-01-01T00:00:00Z',
          input_hashes={k:sha256_file(v) for k,v in paths.items()},config={'frozen_scorer_artifacts':{k:paths[k] for k in FROZEN_ARTIFACTS}})
        state=tmp_path/f'state{round_index}.json';write_manifest(manifest,state)
        output=tmp_path/f'mic{round_index}.pt'
        args=['train']
        for key in ('mic_ledger','peptide_features','peptide_embeddings','peptide_embedding_metadata','expert_means','scaler'):
            args += ['--'+key.replace('_','-'),paths[key]]
        args += ['--state-manifest',str(state),'--current-round',f'R{round_index}','--output',str(output),'--epochs','1']
        if previous:args += ['--parent-checkpoint',str(previous)]
        monkeypatch.setattr('sys.argv',args);main()
        ckpt=torch.load(output,weights_only=True)
        assert ckpt['metadata']['architecture']=='1571->16->1'
        assert ckpt['metadata']['training_config']['learning_rate']==(1e-4 if not round_index else 5e-5)
        assert ckpt['metadata']['frozen_artifact_hashes']=={k:sha256_file(paths[k]) for k in FROZEN_ARTIFACTS}
        previous=output


def test_relative_frozen_paths_resolve_against_state_manifest(tmp_path):
    from src.utils.hash import sha256_file
    keys=set(FROZEN_ARTIFACTS)|{'mic_ledger','peptide_features','peptide_embeddings','peptide_embedding_metadata','expert_means'}
    paths={}
    for key in keys:
        path=tmp_path/key;path.write_text(key);paths[key]=str(path)
    manifest=SimpleNamespace(config={'frozen_scorer_artifacts':{k:k for k in FROZEN_ARTIFACTS}},input_hashes={k:sha256_file(v) for k,v in paths.items()})
    resolved,_=bound_artifacts(manifest,SimpleNamespace(**paths,state_manifest=str(tmp_path/'state.json')))
    assert all(resolved[k]==paths[k] for k in FROZEN_ARTIFACTS)


def test_s12_training_configuration_cannot_be_relabelled_paper():
    for key in ('amp_prior','membrane','target_affinity'):
        config={'epochs':200,'batch_size':32 if key=='target_affinity' else 64,'learning_rate':5e-5 if key=='target_affinity' else 1e-4}
        FrozenSequenceScorer._validate_expert_training(key,{'training_config':config})
        with pytest.raises(ValueError,match='S12'):FrozenSequenceScorer._validate_expert_training(key,{})
        for field in config:
            wrong={**config,field:1}
            with pytest.raises(ValueError,match='S12'):FrozenSequenceScorer._validate_expert_training(key,{'training_config':wrong})
