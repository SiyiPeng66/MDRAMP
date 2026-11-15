"""Rank new prospective proposals, preserve exclusions, then assess haemolytic risk."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from mdramp.calibration_train import fresh_output,sha
from mdramp.prospective_scorer import ProspectiveSequenceScorer
from src.generation.runtime import eligibility_context,assess
from src.external.hemopi_official import score_batch


def diverse_indices(vectors,limit,min_distance):
    vectors=np.asarray(vectors,dtype=float)
    norms=np.linalg.norm(vectors,axis=1)
    if not np.isfinite(vectors).all() or (norms<=0).any():raise ValueError('Invalid diversity vectors')
    unit=vectors/norms[:,None];kept=[]
    for i in range(len(unit)):
        if all(1-float(unit[i]@unit[j])>=min_distance for j in kept):kept.append(i)
        if len(kept)==limit:break
    return kept



def merge_fresh_scores(proposals,scores):
    if scores.sequence.duplicated().any():raise ValueError('Duplicate scorer sequence')
    # Input proposals may already contain stale scores; only the freshly frozen scorer is authoritative.
    clean=proposals.drop(columns=[c for c in scores if c!='sequence' and c in proposals])
    merged=clean.merge(scores,on='sequence',validate='one_to_one')
    if len(merged)!=len(proposals):raise ValueError('Scorer must cover all eligible proposals')
    if not np.isfinite(merged.R_t).all():raise ValueError('Nonfinite R_t')
    return merged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input','scorer-config','reference-sequences','known-sequences','output-dir','hemopi-model','hemopi-official-python'):p.add_argument('--'+key,required=True)
    p.add_argument('--parent');p.add_argument('--parent-family');p.add_argument('--protected-sequences')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--count',type=int,default=30)
    p.add_argument('--min-cosine-distance',type=float,default=.1);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.batch_size<1 or a.count<1 or not 0<=a.min_cosine_distance<=2:raise ValueError('Invalid selection settings')
    import torch
    torch.set_num_threads(a.threads)
    names=['input','scorer_config','reference_sequences','known_sequences']+([ 'parent_family'] if a.parent_family else [])+(['protected_sequences'] if a.protected_sequences else [])
    paths={k:Path(getattr(a,k)) for k in names};hashes={k:sha(v) for k,v in paths.items()};implementation=sha(__file__)
    hemopi_root=Path(a.hemopi_model).resolve()
    safety_paths=[hemopi_root/'hemopi2_classification.py']+[p for folder in ('Model','merci','motif') for p in (hemopi_root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[Path(__file__).resolve().parents[1]/'src/external/hemopi_official.py',Path(__file__).resolve().parents[1]/'src/external/hemopi_rf_entry.py']
    safety_hashes={str(p):sha(p) for p in safety_paths}
    frame=pd.read_csv(a.input)
    if frame.empty or 'sequence' not in frame or frame.sequence.isna().any():raise ValueError('Nonempty proposals required')
    refs=eligibility_context(a,required=True);known=set(pd.read_csv(a.known_sequences).sequence.astype(str));seen=set();rows=[]
    for row in frame.to_dict('records'):
        seq=str(row['sequence']);ok,reason=assess(seq,a.parent,refs)
        if seq in known:ok,reason=False,'previously_tested_exact_sequence'
        elif seq in seen:ok,reason=False,'duplicate_proposal'
        seen.add(seq);rows.append({**row,'sequence':seq,'eligible':ok,'exclusion_reason':reason})
    ledger=pd.DataFrame(rows);valid=ledger[ledger.eligible].copy()
    with fresh_output(a.output_dir) as out:
        ledger.to_csv(out/'proposal_audit.csv',index=False)
        scorer=ProspectiveSequenceScorer(a.scorer_config);scored=[]
        for start in range(0,len(valid),a.batch_size):scored.append(scorer.score_frame(valid.sequence.iloc[start:start+a.batch_size].tolist()))
        if scored:
            scores=pd.concat(scored,ignore_index=True);ranked=merge_fresh_scores(valid,scores).sort_values(['R_t','sequence'],ascending=[False,True]).reset_index(drop=True)
            ranked['antibacterial_rank']=np.arange(1,len(ranked)+1)
            ranked['predicted_adjusted_mic_ug_ml']=np.exp2(ranked.predicted_adjusted_log2_mic)
            if not np.isfinite(ranked.predicted_adjusted_mic_ug_ml).all():raise ValueError('Nonfinite predicted MIC')
            if a.parent:
                chosen=list(range(min(a.count,len(ranked))))
            else:
                vectors=np.stack([scorer.encoder.encode_sequence(str(i),s).global_embedding for i,s in enumerate(ranked.sequence)])
                chosen=diverse_indices(vectors,a.count,a.min_cosine_distance)
            shortlist=ranked.iloc[chosen].copy()
            shortlist['hemopi_relative_risk']=score_batch(shortlist.sequence.tolist(),a.hemopi_model,a.hemopi_official_python)
            if not np.isfinite(shortlist.hemopi_relative_risk).all() or not shortlist.hemopi_relative_risk.between(0,1).all():raise ValueError('Invalid safety score')
            shortlist=shortlist.sort_values(['hemopi_relative_risk','antibacterial_rank']).reset_index(drop=True)
            shortlist['safety_review_order']=np.arange(1,len(shortlist)+1)
            ranked.to_csv(out/'antibacterial_ranked.csv',index=False);shortlist.to_csv(out/'wetlab_candidates.csv',index=False)
            (out/'wetlab_candidates.fasta').write_text(''.join(f'>candidate_{i+1}\n{s}\n' for i,s in enumerate(shortlist.sequence)))
            count=len(shortlist)
        else:
            count=0
            pd.DataFrame(columns=['sequence','R_t','hemopi_relative_risk']).to_csv(out/'wetlab_candidates.csv',index=False)
        scorer.verify_artifacts()
        if any(sha(p)!=h for p,h in safety_hashes.items()):raise ValueError('HemoPI artifacts changed')
        if any(sha(path)!=hashes[k] for k,path in paths.items()) or sha(__file__)!=implementation:raise ValueError('Candidate inputs or implementation changed')
        record={'status':'candidates_ready_for_experimental_review' if count else 'no_eligible_candidates','candidate_count':count,'proposal_count':len(frame),'eligible_count':len(valid),'config':vars(a),'input_sha256':hashes,'scorer_artifact_sha256':scorer.hashes,'hemopi_artifact_sha256':safety_hashes,'implementation_sha256':implementation,'target_semantics':'user_adjusted_approximate_mic','target_divisor':10,'limitations':['Predictions and HemoPI relative-risk scores are not experimental activity or safety confirmation.','Panel is a newly constructed reference proxy, not verified assay-isolate targets.','Local proposals do not use the de novo cosine distance threshold.'],'output_sha256':{x.name:sha(x) for x in out.iterdir() if x.is_file()}}
        (out/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    print(json.dumps({'eligible':len(valid),'candidates':count}))

if __name__=='__main__':main()
