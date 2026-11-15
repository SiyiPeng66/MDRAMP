import pandas as pd
import pytest
from src.score.predict import load_candidates,predict,load_config


def test_input_ids_sequences_and_fasta(tmp_path):
    fasta=tmp_path/'a.fa'; fasta.write_text('>b description\nKALWQKALWQ\n>a\nRALWQKALWQ\n')
    assert load_candidates(fasta).peptide_id.tolist()==['b','a']
    fasta.write_text('>a\nKALWQKALWQ\n>a\nRALWQKALWQ\n')
    with pytest.raises(ValueError,match='unique'):load_candidates(fasta)
    fasta.write_text('>a\nKALWQXALWQ\n')
    with pytest.raises(ValueError,match='standard'):load_candidates(fasta)


def test_prediction_global_ranking_and_duplicate_sequences():
    class Scorer:
        def score_frame(self,seqs):return pd.DataFrame({'sequence':seqs,'R_t':[.5]*len(seqs)})
    frame=pd.DataFrame({'peptide_id':['b','a','c'],'sequence':['KALWQKALWQ']*3})
    result=predict(frame,Scorer(),batch_size=1)
    assert result.peptide_id.tolist()==['a','b','c']
    assert result['rank'].tolist()==[1,2,3]


def test_retained_scorer_pair_chunk_and_batch_invariance():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from src.score.frozen_scorer import FrozenSequenceScorer
    from src.features.physicochemical import compute_descriptors
    from src.features.scaler import StateFeatureScaler
    from src.models.ensemble import ExpertEnsemble
    from src.models.expert_modules import PaperAMPPriorExpert,FeaturewiseGatedMembraneExpert,ProjectedPairAffinityExpert
    from src.models.mic_regressor import MICRegressionHead
    torch.manual_seed(5)
    scorer=object.__new__(FrozenSequenceScorer);scorer.config={'pair_batch_size':2};scorer.hashes={};scorer.fingerprints={};scorer.cache={};scorer.details={}
    class Encoder:
        def encode_sequence(self,identifier,sequence):return SimpleNamespace(global_embedding=np.arange(8,dtype=np.float32)/8)
    scorer.encoder=Encoder()
    columns=list(compute_descriptors('KALWQKALWQ'))
    scorer.scaler=StateFeatureScaler(np.zeros(32),np.ones(32),columns)
    scorer.prior=ExpertEnsemble(lambda:PaperAMPPriorExpert(40),output_key='probability')
    for m in scorer.prior.members:m.fit_support(torch.randn(20,8),list(range(20)))
    scorer.membrane=ExpertEnsemble(lambda:FeaturewiseGatedMembraneExpert(8),output_key='primary')
    scorer.interaction=ExpertEnsemble(lambda:ProjectedPairAffinityExpert(8,8))
    scorer.mic=MICRegressionHead(8,32)
    for model in (scorer.prior,scorer.membrane,scorer.interaction,scorer.mic):model.eval()
    scorer.targets=torch.randn(7,8);scorer.target_ids=[str(i) for i in range(7)]
    sequences=['KALWQKALWQ','RALWQKALWQ']
    batch=scorer.score_frame(sequences)
    scorer.cache={};scorer.details={};scorer.config['pair_batch_size']=7
    singles=pd.concat([scorer.score_frame([s]) for s in sequences])
    np.testing.assert_allclose(batch.R_t,singles.R_t,rtol=1e-5)


def test_formal_config_requires_physical_esm_checkpoint(tmp_path):
    import json
    from src.score.predict import ARTIFACT_KEYS
    config=tmp_path/'scorer.json'
    config.write_text(json.dumps({key:key+'.bin' for key in ARTIFACT_KEYS if key!='esm_checkpoint'}))
    with pytest.raises(ValueError,match='esm_checkpoint'):load_config(config)
    config.write_text(json.dumps({key:key+'.bin' for key in ARTIFACT_KEYS}))
    resolved=load_config(config)
    assert resolved['esm_checkpoint']==str(tmp_path/'esm_checkpoint.bin')
