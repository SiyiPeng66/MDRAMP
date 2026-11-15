import json
import pytest
from src.train.train_target_affinity import validate_curation, verify_curation_evidence
from src.utils.hash import sha256_file


def fixture(tmp_path):
    pairs=tmp_path/'train_pairs.csv';pairs.write_text('pair_id,split,label_type\na,train,background\np,train,positive\n')
    scores=tmp_path/'scores.csv';scores.write_text('pair_id,s_aff_pair\na,0.1\n')
    weights=tmp_path/'weights.pt';weights.write_bytes(b'checkpoint')
    manifest={'schema_version':1,'v6_pair_curation_complete':True,
              'output_sha256':{'train_pairs.csv':sha256_file(pairs)},
              'teacher_screening':{'status':'completed','high_confidence_threshold':.8,
                 'scores':{'path':'scores.csv','sha256':sha256_file(scores)},
                 'weights':{'path':'weights.pt','sha256':sha256_file(weights)}}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    return pairs,path,manifest


def test_default_requires_curation_and_benchmark_truth(tmp_path):
    with pytest.raises(ValueError,match='requires'):validate_curation(tmp_path/'pairs.csv')
    metadata,evidence=validate_curation(tmp_path/'pairs.csv',data_mode='public-benchmark')
    assert metadata['v6_pair_curation_complete'] is False
    assert metadata['teacher_screening']['status']=='not_performed'
    assert evidence=={}


def test_complete_evidence_and_changed_after_training(tmp_path):
    pairs,path,_=fixture(tmp_path)
    metadata,evidence=validate_curation(pairs,path)
    assert metadata['data_mode']=='strict-v6' and len(evidence)==4
    verify_curation_evidence(evidence)
    (tmp_path/'weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):verify_curation_evidence(evidence)
    with pytest.raises(ValueError,match='weights'):validate_curation(pairs,path)


@pytest.mark.parametrize('change',['incomplete','threshold','scores_missing','pair_hash','different_path'])
def test_reject_invalid_screening(tmp_path,change):
    pairs,path,manifest=fixture(tmp_path)
    if change=='incomplete':manifest['v6_pair_curation_complete']=False
    if change=='threshold':manifest['teacher_screening']['high_confidence_threshold']=.9
    if change=='scores_missing':(tmp_path/'scores.csv').unlink()
    if change=='pair_hash':pairs.write_text('changed')
    if change=='different_path':
        pairs=tmp_path/'copied.csv';pairs.write_text((tmp_path/'train_pairs.csv').read_text())
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):validate_curation(pairs,path)


@pytest.mark.parametrize('score_csv',[
    'pair_id,s_aff_pair\na,0.8\n',
    'pair_id,s_aff_pair\na,1.0\n',
    'pair_id,s_aff_pair\na,NaN\n',
    'pair_id,s_aff_pair\na,inf\n',
    'pair_id,s_aff_pair\na,-0.01\n',
    'pair_id,s_aff_pair\nother,0.1\n',
    'pair_id,s_aff_pair\na,0.1\na,0.2\n',
])
def test_strict_verifies_background_scores_even_with_valid_hashes(tmp_path,score_csv):
    pairs,path,manifest=fixture(tmp_path)
    scores=tmp_path/'scores.csv';scores.write_text(score_csv)
    manifest['teacher_screening']['scores']['sha256']=sha256_file(scores)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):validate_curation(pairs,path)


@pytest.mark.parametrize('value',[0.,0.799999])
def test_background_bounds_and_positive_without_teacher(tmp_path,value):
    pairs,path,manifest=fixture(tmp_path)
    scores=tmp_path/'scores.csv';scores.write_text(f'pair_id,s_aff_pair\na,{value}\n')
    manifest['teacher_screening']['scores']['sha256']=sha256_file(scores)
    path.write_text(json.dumps(manifest))
    metadata,_=validate_curation(pairs,path)
    assert metadata['v6_pair_curation_complete'] is True
