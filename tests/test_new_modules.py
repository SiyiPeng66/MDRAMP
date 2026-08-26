import numpy as np
import pandas as pd
import torch

from src.models.ensemble import ExpertEnsemble
from src.models.evidential import evidential_summary, fuse_discounted_evidence
from src.models.expert_modules import ProjectedPairAffinityExpert
from src.models.mdramp_system import MDRAMPSystem
from src.models.expert_modules import AMPPriorExpert, MembraneMechanismExpert
from src.models.gated_integrator import GatedIntegrator
from src.selection.filters import enforce_pairwise_cosine
from src.selection.mixed_policy import select_mixed


def test_ensemble_has_fixed_members_and_statistics():
    ensemble = ExpertEnsemble(lambda: torch.nn.Linear(2, 1))
    result = ensemble(torch.zeros(3, 2))
    assert result["members"].shape[0] == 5
    assert result["mean"].shape == result["std"].shape


def test_projected_pair_expert_shape():
    model = ProjectedPairAffinityExpert(8, 6, physchem_dim=4)
    output = model(torch.zeros(2, 8), torch.zeros(2, 6), torch.zeros(2, 4))
    assert output.shape == (2,)


def test_evidential_outputs_are_bounded():
    result = evidential_summary(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.all((result["probability"] >= 0) & (result["probability"] <= 1))
    fused = fuse_discounted_evidence(np.ones((3, 2)), np.zeros((3, 2)))
    assert np.all(fused["reliability"] > 0)


def test_pairwise_cosine_filter_and_mixed_selection():
    selected = enforce_pairwise_cosine(["A", "B", "C"], np.eye(3), minimum=0.1)
    assert selected == [0, 1, 2]
    frame = pd.DataFrame({
        "selection_category": ["exploitation", "uncertainty", "diversity", "novelty"],
        "R_t": [0.9, 0.8, 0.7, 0.6],
    })
    assert len(select_mixed(frame, budget=4)) == 4


def test_deployed_system_separates_activity_from_consistency():
    prior = AMPPriorExpert(input_dim=3)
    membrane = MembraneMechanismExpert(input_dim=3)
    system = MDRAMPSystem(prior, membrane, None, GatedIntegrator())
    result = system.score_deployed(torch.zeros(2, 3), torch.full((2,), 0.5), torch.tensor([5.0, 6.0]))
    assert "p_cons" in result and "A_t" in result and "R_t" in result
