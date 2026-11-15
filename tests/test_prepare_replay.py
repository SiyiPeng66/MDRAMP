import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
from src.data.prepare_replay import prepare, preserve_split
from src.utils.hash import sha256_file


def fixture(root):
    (root/'data/rawdata/APD').mkdir(parents=True)
    (root/'data/target').mkdir(parents=True)
    (root/'outputs/processed').mkdir(parents=True)
    (root/'data/rawdata/APD/public.fa').write_text('>a first\nKALWQKALWQ\n>b repeat\nKALWQKALWQ\n>c second\nACDEFGHIKL\n>bad\nXXXX\n')
    (root/'data/target/target.fa').write_text('>t1\nACDEFGHIKL\n>t2\nACDEFGHIKL\n')
    pd.DataFrame({'sequence':['ACDEFGHIKL','KALWQKALWQ'],'cluster_id':['C9','C2'],'split':['test','train']}).to_csv(root/'outputs/processed/public_cluster_split.csv',index=False)


def test_public_inventory_preserves_records_splits_and_hashes(tmp_path):
    fixture(tmp_path); out=tmp_path/'prepared';manifest=prepare(tmp_path,out)
    assert manifest['training_ready'] is False
    assert manifest['historical_split_unverified'] is True
    assert manifest['no_final_target_panel'] is True
    assert manifest['counts']['public']['raw_records']==4
    assert manifest['counts']['public']['valid_records']==3
    assert manifest['counts']['public']['deduplicated_sequences']==2
    frame=pd.read_csv(out/'public_sequences.csv');records=pd.read_csv(out/'public_records.csv')
    assert len(records)==4 and records.source_record_id.tolist()==['a','b','c','bad']
    assert frame.set_index('sequence').loc['KALWQKALWQ','cluster_id']=='C2'
    assert frame.set_index('sequence').loc['KALWQKALWQ','peptide_id']==hashlib.sha256(b'KALWQKALWQ').hexdigest()[:16]
    assert not (out/'R0_sequences.csv').exists()
    assert all('private' not in key for key in manifest['input_sha256'])
    for name,digest in manifest['output_sha256'].items(): assert sha256_file(out/name)==digest
    before=sha256_file(out/'manifest.json')
    with pytest.raises(FileExistsError):prepare(tmp_path,out)
    assert sha256_file(out/'manifest.json')==before


def test_mismatch_does_not_publish_or_leave_temporary_files(tmp_path):
    fixture(tmp_path)
    history=tmp_path/'outputs/processed/public_cluster_split.csv'
    pd.read_csv(history).iloc[:1].to_csv(history,index=False)
    with pytest.raises(ValueError,match='set mismatch'):prepare(tmp_path,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()
    assert not list(tmp_path.glob('.prepared*'))

@pytest.mark.parametrize('mode',['duplicate','cluster_leak','invalid_split','missing'])
def test_invalid_history_rejected(mode):
    new=pd.DataFrame({'sequence':['ACDEFGHIKL','KALWQKALWQ']})
    old=new.assign(cluster_id=['a','b'],split=['train','test'])
    if mode=='duplicate':old=pd.concat([old,old.iloc[:1]])
    elif mode=='cluster_leak':old['cluster_id']='a'
    elif mode=='invalid_split':old.loc[0,'split']='validation2'
    else:old.loc[0,'cluster_id']=None
    with pytest.raises(ValueError):preserve_split(new,old)


def test_private_opt_in_is_sequence_only(tmp_path):
    fixture(tmp_path);(tmp_path/'data/private').mkdir();(tmp_path/'data/private/R0.txt').write_text('KALWQKALWQ\nXXXX\n')
    out=tmp_path/'prepared';manifest=prepare(tmp_path,out,include_private=True)
    frame=pd.read_csv(out/'R0_sequences.csv')
    assert len(frame)==1 and frame.data_mode.iloc[0]=='sequence_only'
    assert not {'mic','label','mic_log2'} & set(frame)
    assert manifest['counts']['R0']['raw_records']==2
