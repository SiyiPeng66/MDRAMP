"""Real small neural checkpoints and synthetic cached embeddings; no large ESM load."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from src.features.physicochemical import compute_descriptors
from src.features.scaler import StateFeatureScaler
from src.models.ensemble import ExpertEnsemble
from src.models.expert_modules import PaperAMPPriorExpert,FeaturewiseGatedMembraneExpert,ProjectedPairAffinityExpert
from src.score.evaluate_public import evaluate


def fixture(tmp_path):
    torch.set_num_threads(1);rng=np.random.default_rng(2);seeds=(11,23,37,51,73)
    def features(name,ids,seqs,splits):
        frame=pd.DataFrame([{'peptide_id':pid,'sequence':seq,'split':split,**compute_descriptors(seq)} for pid,seq,split in zip(ids,seqs,splits)])
        path=tmp_path/(name+'.csv');frame.to_csv(path,index=False);return path,frame
    public_path,public=features('public',['p1','p2','p3'],['ACDEFGHIKL','KALWQKALWQ','GIGAVLKVLTT'],['test','train','test'])
    pair_path,pair=features('pair',['q1','q2'],['ACDEFGHIKL','KALWQKALWQ'],['test','test'])
    def embeddings(name,ids,seqs):
        npz=tmp_path/(name+'.npz');meta=tmp_path/(name+'-meta.csv')
        # Deliberately reverse metadata rows; alignment must use record_id/global_index.
        np.savez(npz,global_embeddings=rng.normal(size=(len(ids),8)).astype(np.float32))
        pd.DataFrame({'record_id':ids,'sequence':seqs,'global_index':range(len(ids))}).iloc[::-1].to_csv(meta,index=False)
        return npz,meta
    pe,pm=embeddings('public',public.peptide_id,public.sequence)
    qe,qm=embeddings('pair',pair.peptide_id,pair.sequence)
    te,tm=embeddings('target',['t1','t2'],['ACDEFGHIKLMNPQ','KALWQKALWQKALWQ'])
    columns=[c for c in public if c.startswith('pc_')];scaler=tmp_path/'scaler.json'
    StateFeatureScaler.fit(public[columns].to_numpy(),columns).save(scaler)
    prior=ExpertEnsemble(lambda:PaperAMPPriorExpert(40),seeds,output_key='probability')
    refs=torch.randn(12,8)
    for member in prior.members:member.fit_support(refs,[f'c{i}' for i in range(12)])
    membrane=ExpertEnsemble(lambda:FeaturewiseGatedMembraneExpert(8,32),seeds,output_key='primary')
    interaction=ExpertEnsemble(lambda:ProjectedPairAffinityExpert(8,8,32),seeds)
    def save(name,model,metadata):
        path=tmp_path/(name+'.pt');torch.save({'state_dict':model.state_dict(),'metadata':metadata},path);return path
    priorpath=save('prior',prior,{'architecture':'PaperAMPPriorExpert','seeds':list(seeds),'input_dim':40,'branch_weight':.25,'output_key':'probability'})
    membrane_path=save('membrane',membrane,{'architecture':'FeaturewiseGatedMembraneExpert','seeds':list(seeds),'esm_dim':8,'physchem_dim':32,'output_key':'primary'})
    interaction_path=save('interaction',interaction,{'architecture':'ProjectedPairAffinityExpert','seeds':list(seeds),'peptide_dim':8,'target_dim':8,'physchem_dim':32})
    pairs=tmp_path/'test-pairs.csv'
    pd.DataFrame({'peptide_id':['q1','q2'],'target_id':['t1','t2'],'sequence':pair.sequence,'target_sequence':['ACDEFGHIKLMNPQ','KALWQKALWQKALWQ'],'split':['test','test'],'label':[1,0],'label_type':['positive','background']}).to_csv(pairs,index=False)
    return argparse.Namespace(public_features=str(public_path),public_embeddings=str(pe),public_embedding_metadata=str(pm),amp_prior=str(priorpath),membrane=str(membrane_path),scaler=str(scaler),pair_scores=str(pairs),pair_features=str(pair_path),pair_embeddings=str(qe),pair_embedding_metadata=str(qm),target_embeddings=str(te),target_embedding_metadata=str(tm),interaction_checkpoint=str(interaction_path),output_dir=str(tmp_path/'evaluation'),batch_size=1,device='cpu')


def test_full_cli_real_models_heldout_only_no_mic(tmp_path):
    args=fixture(tmp_path)
    command=[sys.executable,'-m','src.score.evaluate_public']
    for key,value in vars(args).items():command.extend(['--'+key.replace('_','-'),str(value)])
    process=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
    out=Path(args.output_dir);public=pd.read_csv(out/'public_test_expert_scores.csv');pairs=pd.read_csv(out/'pair_test_scores.csv')
    assert public.peptide_id.tolist()==['p1','p3']
    assert len(pairs)==2 # Never Cartesian product of peptide and target caches.
    assert not {'R_t','A_t','predicted_log2_mic'} & set(public)
    assert not {'R_t','A_t','predicted_log2_mic'} & set(pairs)
    metrics=json.loads((out/'metrics.json').read_text())
    assert metrics['public_test']['classification_metrics'] is None
    assert 0<=metrics['pair_test']['roc_auc']<=1
    assert 0<=metrics['pair_test']['average_precision']<=1
    assert 'operational_background_not_confirmed_nonbinding' in set(pairs.evaluation_label_semantics)
    with pytest.raises(FileExistsError):evaluate(args)

@pytest.mark.parametrize('violation',['train_pairs','train_features','wrong_sequence','duplicate_metadata'])
def test_rejects_training_or_misaligned_inputs(tmp_path,violation):
    args=fixture(tmp_path)
    if violation=='train_pairs':
        frame=pd.read_csv(args.pair_scores);frame.loc[0,'split']='train';frame.to_csv(args.pair_scores,index=False)
    elif violation=='train_features':
        frame=pd.read_csv(args.pair_features);frame.loc[0,'split']='train';frame.to_csv(args.pair_features,index=False)
    elif violation=='wrong_sequence':
        frame=pd.read_csv(args.pair_embedding_metadata);frame.loc[0,'sequence']='WRONG';frame.to_csv(args.pair_embedding_metadata,index=False)
    else:
        frame=pd.read_csv(args.public_embedding_metadata);frame.loc[0,'record_id']=frame.loc[1,'record_id'];frame.to_csv(args.public_embedding_metadata,index=False)
    with pytest.raises(ValueError):evaluate(args)
    assert not Path(args.output_dir).exists()
