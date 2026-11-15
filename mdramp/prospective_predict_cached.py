"""Predict adjusted best-panel MIC from audited frozen expert/ESM features."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mdramp import calibration_train as core
from mdramp.prospective_calibration import design,SEMANTICS


def validate_checkpoint(checkpoint,manifest):
    expected={'task':'prospective_adjusted_best_mic','target_semantics':SEMANTICS,'target_divisor':10.,'input_dim':1571,'hidden_dim':16,'output_dim':1,'activation':'GELU','model_kind':'prospective_1571','fold':'all','historical_reproduction':False}
    if any(checkpoint.get(k)!=v for k,v in expected.items()):raise ValueError('Expected frozen final adjusted 1571->16->1 head')
    if checkpoint.get('frozen_artifact_sha256')!=manifest['frozen_artifact_sha256']:raise ValueError('Checkpoint expert/panel bindings differ')
    if checkpoint.get('mean')!=[0.]*1571 or checkpoint.get('scale')!=[1.]*1571:raise ValueError('Unexpected refit feature normalization')


def run(a):
    names=['checkpoint','sequences','features','embeddings','embedding_metadata','expert_means','expert_manifest','scaler']
    names += [n for n in ['source_rows','oof_predictions'] if getattr(a,n,None)]
    paths,hashes=core._identity(a,names);paths['implementation']=str(Path(__file__).resolve());hashes['implementation']=core.sha(__file__)
    import mdramp.prospective_calibration as trainer
    paths['design_helper']=str(Path(trainer.__file__).resolve());hashes['design_helper']=core.sha(trainer.__file__)
    paths['core_helper']=str(Path(core.__file__).resolve());hashes['core_helper']=core.sha(core.__file__)
    seq=pd.read_csv(a.sequences);seq,x,_,manifest=design(a,seq)
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True);validate_checkpoint(ckpt,manifest)
    model=core.CalibrationMLP(1571,16).to(a.device);model.load_state_dict(ckpt['state_dict']);model.eval()
    pred=core.infer(model,ckpt,x,a)[:,0]
    frame=seq.copy();frame['predicted_adjusted_log2_mic']=pred;frame['predicted_adjusted_mic_ug_ml']=np.exp2(pred.astype(float))
    frame['target_semantics']=SEMANTICS;frame['target_divisor']=10.;frame['input_target_semantics']=ckpt.get('input_target_semantics','measured_mic');frame['state_id']=ckpt['state_id']
    frame['used_in_final_model_training']=frame.peptide_id.isin(ckpt['train_peptide_ids'])
    frame['fixed_temperature_activity_32']=1/(1+np.exp(np.clip(pred-5,-700,700)))
    if not np.isfinite(frame[['predicted_adjusted_log2_mic','predicted_adjusted_mic_ug_ml']]).all().all():raise ValueError('Nonfinite prediction')
    if a.oof_predictions:
        oof=pd.read_csv(a.oof_predictions)
        if oof.peptide_id.duplicated().any() or set(oof.peptide_id)!=set(frame.peptide_id):raise ValueError('OOF coverage mismatch')
        if not oof.target_semantics.eq(SEMANTICS).all() or not oof.target_divisor.eq(10).all():raise ValueError('OOF target semantics mismatch')
        frame=frame.merge(oof[['peptide_id','esm_pc']].rename(columns={'esm_pc':'oof_predicted_adjusted_log2_mic'}),on='peptide_id',validate='one_to_one')
        frame['oof_predicted_adjusted_mic_ug_ml']=np.exp2(frame.oof_predicted_adjusted_log2_mic)
    with core.fresh_output(a.output_dir) as out:
        frame.to_csv(out/'predictions.csv',index=False)
        if a.source_rows:
            rows=pd.read_csv(a.source_rows,usecols=['source_row_id','peptide_id','sequence','round','new_name','original_id'])
            rows=rows[rows.peptide_id.isin(frame.peptide_id)]
            joined=rows.merge(frame,on=['peptide_id','sequence'],validate='many_to_one')
            if len(joined)!=len(rows) or set(joined.peptide_id)!=set(frame.peptide_id):raise ValueError('Source round/sequence alignment mismatch')
            joined.to_csv(out/'source_predictions.csv',index=False)
            unique=joined.drop_duplicates(['round','peptide_id']);stats=[]
            for round_name,g in unique.groupby('round',dropna=False):
                for column in ['predicted_adjusted_mic_ug_ml','oof_predicted_adjusted_mic_ug_ml']:
                    if column not in g:continue
                    v=g[column].to_numpy(float);stats.append({'round':round_name,'prediction_type':'retrospective_OOF' if column.startswith('oof') else 'final_model_training_fit','n_unique_peptides':len(g),'mean':float(v.mean()),'median':float(np.median(v)),'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75)),'min':float(v.min()),'max':float(v.max()),'fraction_le_3_2':float((v<=3.2).mean()),'fraction_le_25_6':float((v<=25.6).mean())})
            pd.DataFrame(stats).to_csv(out/'round_distributions.csv',index=False)
        if any(core.sha(p)!=hashes[k] for k,p in paths.items()):raise ValueError('Prediction inputs changed')
        core._json(out/'manifest.json',{'status':'complete','state_id':ckpt['state_id'],'target_semantics':SEMANTICS,'target_divisor':10,'input_target_semantics':ckpt.get('input_target_semantics','measured_mic'),'record_count':len(frame),'frozen_artifact_sha256':manifest['frozen_artifact_sha256'],'input_paths':paths,'input_sha256':hashes,'config':vars(a),'limitations':['Adjusted labels are user assumptions, not measured MIC improvement.','Final model predictions on its 230 training peptides are fitted values, not generalization evidence.','OOF results are retrospective cluster-separated internal validation; R0-R5 distributions are descriptive, not prospective round validation.','Fixed temperature sigmoid is not an independently calibrated probability.'],'output_sha256':{p.name:core.sha(p) for p in out.iterdir() if p.is_file()}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['checkpoint','sequences','features','embeddings','embedding-metadata','expert-means','expert-manifest','scaler','output-dir']:p.add_argument('--'+n,required=True)
    p.add_argument('--source-rows');p.add_argument('--oof-predictions');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if min(a.batch_size,a.threads)<1:raise ValueError('Positive batch/threads required')
    torch.set_num_threads(a.threads);run(a)

if __name__=='__main__':main()
