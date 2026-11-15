import numpy as np
import torch
from src.models.expert_modules import ProjectedPairAffinityExpert
from mdramp.panel_pair_cache import projected_logits,atomic_npz


def test_accelerated_actual_pair_head_matches_original():
    torch.manual_seed(11)
    model=ProjectedPairAffinityExpert(1536,1536,32).eval()
    peptide=torch.randn(3,1536);targets=torch.randn(17,1536);pc=torch.randn(3,32)
    with torch.inference_mode():
        pp=model.peptide_norm(model.peptide_projection(peptide));tp=model.target_norm(model.target_projection(targets))
        for i in range(3):
            fast=projected_logits(model,pp[i:i+1],tp,pc[i:i+1])
            original=model.pair_logits(peptide[i:i+1].expand(17,-1),targets,pc[i:i+1].expand(17,-1))
            assert float((fast-original).abs().max())<=1e-5
            torch.testing.assert_close(fast,original,atol=1e-5,rtol=1e-5)


def test_atomic_probe_cache_is_readable_and_no_temp_left(tmp_path):
    path=tmp_path/'one.npz';values=np.arange(30,dtype=np.float32).reshape(5,6)
    atomic_npz(path,logits=values,fixed=np.ones(4))
    with np.load(path) as saved:np.testing.assert_array_equal(saved['logits'],values)
    assert [p.name for p in tmp_path.iterdir()]==['one.npz']


def test_cache_identity_shape_and_finite_defend_resume_and_assembly(tmp_path):
    import json
    import pytest
    from src.utils.hash import sha256_file
    from mdramp.panel_pair_cache import validate_probe_cache
    path=tmp_path/'000000.npz'
    def write(logits,fixed,record_id='expected'):
        atomic_npz(path,logits=logits,fixed=fixed)
        path.with_suffix('.json').write_text(json.dumps({'record_id':record_id,'sha256':sha256_file(path)}))
    write(np.zeros((5,6)),np.zeros(4));validate_probe_cache(path,'expected',6)
    write(np.zeros((5,6)),np.zeros(4),'another-probe')
    with pytest.raises(ValueError,match='identity'):validate_probe_cache(path,'expected',6)
    for logits,fixed in [(np.zeros((6,5)),np.zeros(4)),(np.zeros((5,6)),np.zeros(5))]:
        write(logits,fixed)
        with pytest.raises(ValueError,match='shape'):validate_probe_cache(path,'expected',6)
    for logits,fixed in [(np.full((5,6),np.nan),np.zeros(4)),(np.zeros((5,6)),np.full(4,np.inf))]:
        write(logits,fixed)
        with pytest.raises(ValueError,match='Nonfinite'):validate_probe_cache(path,'expected',6)


def test_dependencies_are_module_relative_not_working_directory(tmp_path,monkeypatch):
    from mdramp.panel_pair_cache import dependency_paths
    expected=dependency_paths();monkeypatch.chdir(tmp_path)
    assert dependency_paths()==expected
    assert all(p.is_absolute() and p.is_file() for p in expected)
    assert {'scaler.py','ensemble.py','score_candidates.py','embedding_store.py'} <= {p.name for p in expected}
