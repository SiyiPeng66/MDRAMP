import types
import numpy as np
import pandas as pd
import pytest
import torch
from src.external.tpeppro_public import score_encoded,validate_record
from src.external.tpeppro_runtime import TPepProRuntime
from src.data.screen_public_pairs import screen_pairs


def test_cached_head_matches_original_forward():
    model=TPepProRuntime.__new__(TPepProRuntime);torch.nn.Module.__init__(model)
    model.fc1=torch.nn.Linear(256,512);model.BatchNorm1d_1=torch.nn.BatchNorm1d(512)
    model.fc2=torch.nn.Linear(512,256);model.BatchNorm1d_2=torch.nn.BatchNorm1d(256);model.out=torch.nn.Linear(256,1);model.eval()
    model.encode=types.MethodType(lambda self,emb,contact:emb,model)
    receptor=torch.randn(1,128);peptide=torch.randn(1,128)
    assert score_encoded(model,receptor,peptide)==model(receptor,None,peptide,None)


def test_contacts_must_match_exact_sequence(tmp_path):
    np.savez(tmp_path/'p.npz',seq='ACDE',contact=np.eye(4))
    embeddings={'p':np.ones((4,1024),np.float32)}
    assert validate_record('p',{'p':'ACDE'},embeddings,tmp_path)[0].shape==(4,1024)
    with pytest.raises(ValueError,match='sequence_mismatch'):validate_record('p',{'p':'ACDF'},embeddings,tmp_path)


def test_screen_all_rows_high_missing_and_invalid_accounted():
    rows=[];scores=[]
    for split in ['train','validation','test']:
        for label in ['positive','background']:
            for i in range(4):
                identifier=f'{split}-{label}-{i}'
                rows.append(dict(pair_id=identifier,peptide_id=identifier,target_id='t',sequence='ACDEFGHIKL',target_sequence='ACDEFGHIKLMN',label_type=label,explicit_label=1 if label=='positive' else '',cluster_id=split,split=split,source_action_rows=str(i)))
                if label=='background' and i!=3:scores.append(dict(pair_id=identifier,s_aff_pair=[.2,.8,np.nan][i]))
    selected,excluded=screen_pairs(pd.DataFrame(rows),pd.DataFrame(scores))
    assert len(selected)==6 and len(excluded)==18
    assert (selected[selected.label_type=='background'].teacher_score==.2).all()
    assert selected.loc[selected.label_type=='positive','s_aff_pair'].isna().all()
    assert selected.loc[selected.label_type=='positive','teacher_score'].isna().all()
    assert (selected.loc[selected.label_type=='positive','explicit_label']==1).all()
    assert 'source_action_rows' in selected
    assert set(excluded.screen_exclusion_reason)=={'high_confidence_teacher_positive','invalid_input_nonfinite_or_out_of_range_teacher_score','invalid_input_missing_teacher_score','post_screen_balance_surplus'}
    with pytest.raises(ValueError,match='Unique teacher'):screen_pairs(pd.DataFrame(rows),pd.concat([pd.DataFrame(scores),pd.DataFrame(scores)]))


@pytest.mark.parametrize("inject_failure",[False,True])
def test_screen_cli_emits_strict_hash_bound_manifest(tmp_path,monkeypatch,inject_failure):
    import json,sys
    from src.data import screen_public_pairs as module
    from src.utils.hash import sha256_file
    rows=[];scores=[]
    for split in ['train','validation','test']:
        for label in ['positive','background']:
            key=f'{split}_{label}'
            rows.append(dict(pair_id=key,peptide_id=key,target_id='t',sequence='ACDEFGHIKL',target_sequence='ACDEFGHIKLMN',label_type=label,explicit_label=1 if label=='positive' else '',cluster_id=split,split=split))
            if label=='background':scores.append({'pair_id':key,'s_aff_pair':.2,'status':'scored'})
    source=tmp_path/'all_pairs.csv';pd.DataFrame(rows).to_csv(source,index=False)
    scorepath=tmp_path/'scores.csv';pd.DataFrame(scores).to_csv(scorepath,index=False)
    weights=tmp_path/'weights';weights.write_bytes(b'fixture only')
    pd.DataFrame(columns=['pair_id','reason']).to_csv(tmp_path/'rejections.csv',index=False)
    structures=tmp_path/'structure.json';structures.write_text(json.dumps({'status':'complete','binding':{'input_sha256':{'pairs':sha256_file(source)}},'output_sha256':{'rejections.csv':sha256_file(tmp_path/'rejections.csv')}}))
    t5=tmp_path/'t5.json';t5.write_text('{}')
    public=tmp_path/'public.json';public.write_text(json.dumps({'output_sha256':{'all_pairs.csv':sha256_file(source)}}))
    teacher=tmp_path/'teacher.json';teacher.write_text(json.dumps({'scores':{'sha256':sha256_file(scorepath)},'inputs':{key:{'path':str(path),'sha256':sha256_file(path)} for key,path in [('checkpoint',weights),('structure_manifest',structures),('embedding_manifest',t5)]}}))
    out=tmp_path/'screened'
    monkeypatch.setattr(sys,'argv',['program','--all-pairs',str(source),'--scores',str(scorepath),'--weights',str(weights),'--teacher-manifest',str(teacher),'--public-manifest',str(public),'--output-dir',str(out)])
    if inject_failure:
        original_write=module.atomic_write
        def fail_after_partial_write(*args,**kwargs):
            original_write(*args,**kwargs)
            raise OSError('injected disk failure')
        with monkeypatch.context() as patch:
            patch.setattr(module,'atomic_write',fail_after_partial_write)
            with pytest.raises(OSError,match='injected disk failure'):module.main()
        assert not out.exists()
        assert not list(tmp_path.glob('.screened.staging-*'))
    module.main()
    positive=pd.read_csv(out/'all_pairs.csv').query("label_type == 'positive'")
    assert positive.s_aff_pair.isna().all() and positive.teacher_score.isna().all()
    assert (positive.explicit_label==1).all()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['v6_pair_curation_complete'] is True
    assert manifest['teacher_screening']['high_confidence_threshold']==.8
    for name,digest in manifest['output_sha256'].items():assert sha256_file(out/name)==digest
