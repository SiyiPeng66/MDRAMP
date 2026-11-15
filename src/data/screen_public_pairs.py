"""Strict V6 teacher exclusion and deterministic per-split public-pair balancing."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import fcntl
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.prott5_embeddings import atomic_json,atomic_write
from src.utils.hash import sha256_file


def screen_pairs(pairs,scores,seed=11,invalid_input_reasons=None):
    required={'pair_id','peptide_id','target_id','sequence','target_sequence','label_type','explicit_label','cluster_id','split'}
    if not required.issubset(pairs) or pairs.pair_id.duplicated().any():raise ValueError('Unique original pair IDs and complete sequence/source columns required')
    if not {'pair_id','s_aff_pair'}.issubset(scores) or scores.pair_id.duplicated().any():raise ValueError('Unique teacher pair_id and s_aff_pair required')
    if set(scores.pair_id)-set(pairs.pair_id):raise ValueError('Teacher contains unknown original pair IDs')
    lookup=scores.set_index('pair_id');eligible=[];excluded=[]
    for record in pairs.to_dict('records'):
        reason=None;record['teacher_score']=np.nan
        if record['split'] not in {'train','validation','test'}:reason='original_split_quarantine'
        elif record['label_type']=='positive':
            if float(record['explicit_label'])!=1:raise ValueError('Positive pair must have explicit label 1')
            record['s_aff_pair']=np.nan
        elif record['label_type']=='background':
            if record['pair_id'] not in lookup.index:
                if invalid_input_reasons is not None and record['pair_id'] not in invalid_input_reasons:raise ValueError('Background lacks both score and explicit structural rejection')
                reason='invalid_input:'+invalid_input_reasons[record['pair_id']] if invalid_input_reasons is not None else 'invalid_input_missing_teacher_score'
            else:
                score_row=lookup.loc[record['pair_id']];value=pd.to_numeric(score_row['s_aff_pair'],errors='coerce')
                if 'status' in score_row and score_row['status']!='scored':reason='invalid_input:'+str(score_row.get('reason','teacher_status'))
                elif not np.isfinite(value) or not 0<=value<=1:reason='invalid_input_nonfinite_or_out_of_range_teacher_score'
                elif value>=.8:reason='high_confidence_teacher_positive'
                else:record['teacher_score']=float(value);record['s_aff_pair']=float(value)
        else:raise ValueError('Unknown original label_type')
        if reason:record['screen_exclusion_reason']=reason;excluded.append(record)
        else:eligible.append(record)
    valid=pd.DataFrame(eligible);selected=[]
    if valid.empty:raise ValueError('No eligible public pairs')
    for split in ['train','validation','test']:
        frame=valid[valid.split==split].copy();pos=frame[frame.label_type=='positive'];bg=frame[frame.label_type=='background'];budget=min(len(pos),len(bg))
        if not budget:raise ValueError(f'No nonempty balanced pairs available for {split}')
        for group in [pos,bg]:
            group=group.assign(_order=group.pair_id.map(lambda value:hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest())).sort_values(['_order','pair_id']).drop(columns='_order')
            selected.extend(group.head(budget).to_dict('records'))
            for record in group.iloc[budget:].to_dict('records'):
                record['screen_exclusion_reason']='post_screen_balance_surplus';excluded.append(record)
    selected=pd.DataFrame(selected).sort_values(['split','pair_id']).reset_index(drop=True)
    if len(selected)+len(excluded)!=len(pairs):raise AssertionError('Every source row must be accounted for')
    if (selected.groupby('cluster_id').split.nunique()>1).any():raise ValueError('Cluster crosses output splits')
    for id_col,seq_col in [('peptide_id','sequence'),('target_id','target_sequence')]:
        if (selected.groupby(id_col)[seq_col].nunique()>1).any():raise ValueError('Identifier maps to conflicting sequences')
    return selected,pd.DataFrame(excluded)


@contextmanager
def staged_output_directory(output):
    """Publish a complete new dataset directory or leave no final output."""
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.publish.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('Screened output publication is locked') from error
        if output.exists():raise ValueError('Screened output must use a new versioned directory')
        staging=Path(tempfile.mkdtemp(prefix='.'+output.name+'.staging-',dir=output.parent))
        try:
            yield staging
            if output.exists():raise ValueError('Final screened output appeared during publication')
            os.rename(staging,output)
        finally:
            if staging.exists():shutil.rmtree(staging)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['all-pairs','scores','weights','teacher-manifest','public-manifest','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=11);a=p.parse_args();out=Path(a.output_dir)
    if out.exists():raise ValueError('Screened output must use a new versioned directory')
    inputs={name:{'path':str(Path(getattr(a,name)).resolve()),'sha256':sha256_file(getattr(a,name))} for name in ['all_pairs','scores','weights','teacher_manifest','public_manifest']}
    public=json.loads(Path(a.public_manifest).read_text())
    if public.get('output_sha256',{}).get('all_pairs.csv')!=inputs['all_pairs']['sha256']:raise ValueError('Original public pair manifest mismatch')
    teacher=json.loads(Path(a.teacher_manifest).read_text())
    if teacher.get('scores',{}).get('sha256')!=inputs['scores']['sha256'] or teacher.get('inputs',{}).get('checkpoint',{}).get('sha256')!=inputs['weights']['sha256']:raise ValueError('Teacher scores/weights provenance mismatch')
    for key in ['structure_manifest','embedding_manifest']:
        record=teacher.get('inputs',{}).get(key,{})
        if not record.get('path') or sha256_file(record['path'])!=record.get('sha256'):raise ValueError('Missing/changed structure or ProtT5 provenance')
    pairs=pd.read_csv(a.all_pairs,keep_default_na=False);scores=pd.read_csv(a.scores)
    structure_record=teacher['inputs']['structure_manifest']
    structure=json.loads(Path(structure_record['path']).read_text())
    rejection_path=Path(structure_record['path']).parent/'rejections.csv'
    if structure.get('status')!='complete' or sha256_file(rejection_path)!=structure.get('output_sha256',{}).get('rejections.csv'):raise ValueError('Verified structural rejection ledger required')
    if structure.get('binding',{}).get('input_sha256',{}).get('pairs')!=inputs['all_pairs']['sha256']:raise ValueError('Structures were not derived from this original pair ledger')
    rejections=pd.read_csv(rejection_path)
    if rejections.pair_id.duplicated().any():raise ValueError('Duplicate structural rejection IDs')
    reasons=dict(zip(rejections.pair_id,rejections.reason.astype(str)))
    selected,excluded=screen_pairs(pairs,scores,a.seed,reasons)
    frames={'all_pairs.csv':selected,'exclusions.csv':excluded,
        'peptide_sequences.csv':selected[['peptide_id','sequence','cluster_id','split']].drop_duplicates('peptide_id'),
        'proteins.csv':selected[['target_id','target_sequence']].drop_duplicates('target_id')}
    for split,name in [('train','train_pairs.csv'),('validation','val_pairs.csv'),('test','test_pairs.csv')]:frames[name]=selected[selected.split==split]
    with staged_output_directory(out) as staging:
        for name,frame in frames.items():atomic_write(staging/name,lambda handle,f=frame:f.to_csv(handle,index=False),False)
        for record in inputs.values():
            if sha256_file(record['path'])!=record['sha256']:raise ValueError('Curation input changed')
        for key in ('structure_manifest', 'embedding_manifest'):
            record = teacher['inputs'][key]
            if sha256_file(record['path']) != record['sha256']:
                raise ValueError('Curation secondary provenance changed')
        if sha256_file(rejection_path) != structure['output_sha256']['rejections.csv']:
            raise ValueError('Structural rejection ledger changed during curation')
        manifest={'schema_version':1,'v6_pair_curation_complete':True,'dataset_version':out.name,'historical_31248_reproduction':False,'teacher_screening':{'status':'completed','high_confidence_threshold':.8,'scores':inputs['scores'],'weights':inputs['weights']},'source_inputs':inputs,'structures_provenance':teacher['inputs']['structure_manifest'],'prott5_provenance':teacher['inputs']['embedding_manifest'],'output_sha256':{name:sha256_file(staging/name) for name in frames},'algorithm_sha256':sha256_file(__file__),'seed':a.seed,'source_rows':len(pairs),'retained_rows':len(selected),'excluded_rows':len(excluded),'exclusion_counts':excluded.screen_exclusion_reason.value_counts().to_dict() if len(excluded) else {},'split_counts':selected.groupby(['split','label_type']).size().unstack(fill_value=0).to_dict('index'),'semantics':'Curated explicit positives plus operational background screened to finite teacher score <0.8, not experimentally verified nonbinding.'}
        atomic_json(staging/'manifest.json',manifest)
    print(json.dumps(manifest['split_counts'],indent=2))


if __name__=='__main__':main()
