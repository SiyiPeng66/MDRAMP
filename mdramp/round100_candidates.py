"""Score new proposals and retain historical rows in six explicitly new 100-row sets."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.hash import sha256_file
from src.data.cluster_split import length_aware_identity
from mdramp.calibration_train import fresh_output
COUNTS={'R0':30,'R1':30,'R2':30,'R3':60,'R4':10,'R5':10}


def select_and_allocate(scored,known):
    ranked=scored.sort_values(['R_t','sequence'],ascending=[False,True]).reset_index(drop=True)
    if ranked.sequence.duplicated().any() or set(ranked.sequence)&set(known):raise ValueError('New pool must be unique and exclude all230 originals')
    if len(ranked)<430:raise ValueError(f'Need430 actual novel candidates, only{len(ranked)} available')
    selected=[];deferred=[]
    for i,sequence in enumerate(ranked.sequence):
        similar=any(min(len(sequence),len(ranked.sequence[j]))/max(len(sequence),len(ranked.sequence[j]))>=.9 and length_aware_identity(sequence,ranked.sequence[j])>=.9 for j in selected)
        if similar:deferred.append(i)
        else:selected.append(i)
        if len(selected)==430:break
    diverse=set(selected)
    if len(selected)<430:selected+=deferred[:430-len(selected)]
    chosen=ranked.loc[selected].sort_values(['R_t','sequence'],ascending=[False,True]).copy()
    chosen['diversity_90_representative']=chosen.index.isin(diverse)
    counts={r:0 for r in COUNTS};assign=[]
    for _ in range(len(chosen)):
        round_name=min((r for r in COUNTS if counts[r]<100-COUNTS[r]),key=lambda r:(counts[r]/(100-COUNTS[r]),list(COUNTS).index(r)))
        counts[round_name]+=1;assign.append(round_name)
    chosen['round']=assign;return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['score','assemble'],required=True)
    for name in ('input','scorer-config','source-predictions','source-manifest','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--score-manifest');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from mdramp.prospective_scorer import ProspectiveSequenceScorer
    scorer=ProspectiveSequenceScorer(a.scorer_config)
    paths=[a.input,a.scorer_config,a.source_predictions,a.source_manifest,__file__]+([a.score_manifest] if a.score_manifest else [])
    hashes={str(Path(x).resolve()):sha256_file(x) for x in paths}
    source=pd.read_csv(a.source_predictions);sm=json.loads(Path(a.source_manifest).read_text())
    if sm['output_sha256'].get(Path(a.source_predictions).name)!=sha256_file(a.source_predictions):raise ValueError('Source prediction hash mismatch')
    if sm['input_sha256'].get('checkpoint')!=scorer.hashes['mic_checkpoint'] or sm.get('frozen_artifact_sha256')!={k:v for k,v in scorer.hashes.items() if k!='mic_checkpoint'}:raise ValueError('Original predictions must use this exact final575 model')
    if len(source)!=230 or source.sequence.nunique()!=230:raise ValueError('Expected all230 unique original sequences')
    known=set(source.sequence);frame=pd.read_csv(a.input)
    with fresh_output(a.output_dir) as out:
        if a.stage=='score':
            rows=[];seen=set()
            for row in frame.to_dict('records'):
                sequence=str(row.get('sequence',''));reason=''
                if not 10<=len(sequence)<=50 or any(x not in 'ACDEFGHIKLMNPQRSTVWY' for x in sequence):reason='invalid_sequence'
                elif sequence in known:reason='previously_measured_exact'
                elif sequence in seen:reason='duplicate_generated_proposal'
                seen.add(sequence);rows.append({**row,'sequence':sequence,'exclusion_reason':reason})
            audit=pd.DataFrame(rows);audit.to_csv(out/'proposal_audit.csv',index=False);valid=audit[audit.exclusion_reason==''].copy();chunks=[]
            for start in range(0,len(valid),a.batch_size):
                chunks.append(scorer.score_frame(valid.sequence.iloc[start:start+a.batch_size].tolist()));print(f'actual575 scoring {min(start+a.batch_size,len(valid))}/{len(valid)}',flush=True)
            if not chunks:raise ValueError('No valid novel proposals')
            scores=pd.concat(chunks,ignore_index=True)
            clean=valid.drop(columns=[c for c in scores if c!='sequence' and c in valid])
            result=clean.merge(scores,on='sequence',validate='one_to_one');result['predicted_adjusted_mic_ug_ml']=np.exp2(result.predicted_adjusted_log2_mic)
            result.to_csv(out/'scored_pool.csv',index=False)
        else:
            if not a.score_manifest:raise ValueError('Scored pool manifest required')
            manifest=json.loads(Path(a.score_manifest).read_text())
            if manifest.get('scorer_artifact_sha256')!=scorer.hashes or manifest['output_sha256'].get(Path(a.input).name)!=sha256_file(a.input):raise ValueError('Scored pool bindings differ')
            if not np.isfinite(frame[['R_t','predicted_adjusted_mic_ug_ml']]).all().all():raise ValueError('Nonfinite actual predictions')
            chosen=select_and_allocate(frame,known);chosen['origin']='new_prospective_supplement';chosen['used_in_final_model_training']=False;chosen['historical_reproduction']=False
            chosen['display_name']=[r+'_NEW_'+str(i+1).zfill(3) for i,r in enumerate(chosen['round'])]
            original=source[source['round'].isin(COUNTS)].copy()
            if original.groupby('round').size().to_dict()!=COUNTS:raise ValueError('Historical170 round allocation differs')
            original['origin']='retained_original_measured_sequence';original['display_name']=original.new_name.fillna('').where(original.new_name.fillna('')!='',original.original_id);original['historical_reproduction']=False
            result=pd.concat([original,chosen],ignore_index=True);result['prediction_threshold_le128']=result.predicted_adjusted_mic_ug_ml<=128
            result['threshold_semantics']='user_adjusted_approximate_MIC_not_measured';result['round_semantics']='assembly_group; supplements generated_now_using_one_final575_state'
            result['safety_status']='not_established_by_MIC_prediction'
            if len(result)!=600 or result.sequence.nunique()!=600 or not result.groupby('round').size().eq(100).all():raise ValueError('600 unique round100 invariant failed')
            result.to_csv(out/'all_rounds.csv',index=False);stats=[]
            for round_name,part in result.groupby('round'):
                part.to_csv(out/f'{round_name}.csv',index=False)
                (out/f'{round_name}.fasta').write_text(''.join(f'>{r.display_name}|{r.origin}\n{r.sequence}\n' for r in part.itertuples()))
                stats.append({'round':round_name,'total':len(part),'original_count':int((part.origin=='retained_original_measured_sequence').sum()),'new_count':int((part.origin=='new_prospective_supplement').sum()),'predicted_adjusted_le128':int(part.prediction_threshold_le128.sum()),'new_predicted_adjusted_le128':int(part.loc[part.origin=='new_prospective_supplement','prediction_threshold_le128'].sum())})
            pd.DataFrame(stats).to_csv(out/'round_statistics.csv',index=False)
        scorer.verify_artifacts()
        if any(sha256_file(x)!=v for x,v in hashes.items()):raise ValueError('Inputs changed')
        record={'stage':a.stage,'status':'complete','state_id':scorer.checkpoint['state_id'],'scorer_artifact_sha256':scorer.hashes,'input_sha256':hashes,'record_count':len(result),'target_semantics':'user_adjusted_approximate_mic','target_divisor':10,'historical_reproduction':False,'selection_policy':'Descending actual final575 R_t; first greedy90percent identity diverse representatives, fill deficits from highest ranked deferred with diversity flag false; normalized remaining quota round allocation. No threshold guarantee or silent discarded original.','limitations':['Each round is an assembly group, not a recreation of its historical scoring or generation state.','Original rows have fitted final-model predictions, not prospective validation.','No experimental safety guarantee; <=128 is a recorded adjusted-prediction indicator, not a hard-filled biological success criterion.'],'output_sha256':{x.name:sha256_file(x) for x in out.iterdir() if x.is_file()}}
        (out/'manifest.json').write_text(json.dumps(record,indent=2))
if __name__=='__main__':main()
