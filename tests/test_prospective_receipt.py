"""Independent completion-receipt checks: best checkpoint existence is insufficient."""
import fcntl
import json
import torch
from mdramp.prospective_receipt import inspect
from mdramp.calibration_train import sha


def fixture(tmp_path):
    checkpoint=tmp_path/'best.pt';state=tmp_path/'state.pt'
    payload={'conditional':False,'metadata':{'stage':'supervised_pretraining','epoch':7,'validation_nll':2.,
      'arguments':{'action':'pretrain','epochs':100},'input_hashes':{'data':'source'}}}
    torch.save(payload,checkpoint)
    training={'schema_version':1,'next_epoch':50,'stale':0,'best_sha256':sha(checkpoint),
      'binding':{'patience':10,'conditional':False,'data_sha256':'source','destinations':{'output':str(checkpoint),'state':str(state)}}}
    torch.save(training,state)
    marker=tmp_path/'complete.json';marker.write_text(json.dumps({'status':'paper_expert_ensembles_complete'}))
    spec={'generators':[{'checkpoint':'best.pt','training_state':'state.pt','conditional':False,'max_epochs':100}],
          'artifacts':[],'expert_completion_marker':'complete.json','expert_checkpoints':[]}
    return spec,training,state,checkpoint


def test_unlocked_intermediate_best_is_not_complete(tmp_path):
    spec,training,state,_=fixture(tmp_path)
    assert inspect(spec,tmp_path) is None
    training['stale']=10;torch.save(training,state)
    result=inspect(spec,tmp_path)
    assert result['status']=='completed_external_training_verified'
    assert result['artifact_sha256'][str(state)]==sha(state)


def test_active_training_lock_blocks_even_terminal_state(tmp_path):
    spec,training,state,checkpoint=fixture(tmp_path)
    training['next_epoch']=100;torch.save(training,state)
    with checkpoint.with_name(checkpoint.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert inspect(spec,tmp_path) is None
    assert inspect(spec,tmp_path) is not None


def test_symlink_alias_cannot_bypass_training_lock(tmp_path):
    spec,training,state,checkpoint=fixture(tmp_path)
    training['next_epoch']=100;torch.save(training,state)
    alias=tmp_path/'alias.pt';alias.symlink_to(checkpoint)
    spec['generators'][0]['checkpoint']='alias.pt'
    with checkpoint.with_name(checkpoint.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert inspect(spec,tmp_path) is None


def test_terminal_state_cannot_borrow_another_destination_or_budget(tmp_path):
    import pytest
    spec,training,state,_=fixture(tmp_path);training['next_epoch']=100
    training['binding']['destinations']['output']=str(tmp_path/'other.pt');torch.save(training,state)
    with pytest.raises(ValueError,match='identity'):inspect(spec,tmp_path)
    training['binding']['destinations']['output']=str(tmp_path/'best.pt');torch.save(training,state)
    spec['generators'][0]['max_epochs']=1
    with pytest.raises(ValueError,match='budget'):inspect(spec,tmp_path)


def test_expert_checkpoint_is_bound_even_outside_generic_artifact_list(tmp_path):
    spec,training,state,_=fixture(tmp_path);training['next_epoch']=100;torch.save(training,state)
    spec['expert_checkpoints']=['expert.pt'];assert inspect(spec,tmp_path) is None
    expert=tmp_path/'expert.pt';torch.save({'metadata':{'architecture':'PaperAMPPriorExpert',
       'training_config':{'epochs':200,'batch_size':64},'seeds':[11,23,37,51,73]}},expert)
    result=inspect(spec,tmp_path)
    assert result['artifact_sha256'][str(expert)]==sha(expert)
