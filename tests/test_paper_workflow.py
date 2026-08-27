from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.external.tpeppro_wrapper import aggregate_pair_frame
from src.models.gated_integrator import GateParams, GatedIntegrator, integrate_numpy
from src.models.expert_modules import FeaturewiseGatedMembraneExpert, PaperAMPPriorExpert
from src.optimization.pep_b_genetic import GAConfig, FitnessComponents, PAPER_SEEDS, optimize, optimize_five_runs
from src.train.expert_losses import amp_prior_loss, masked_membrane_loss
from src.selection.freeze_round import freeze_round_selection, next_reserve_candidate


def test_amp_prior_uses_both_nnpu_and_manifold_branches():
    model = PaperAMPPriorExpert(6, branch_weight=0.5)
    positive, unlabelled = model(torch.ones(4, 6)), model(torch.zeros(4, 6))
    loss = amp_prior_loss(positive, unlabelled)
    loss.backward()
    assert model.pu_head.weight.grad is not None
    assert model.manifold_head.weight.grad is not None


def test_missing_membrane_auxiliary_labels_have_no_gradient():
    model = FeaturewiseGatedMembraneExpert(8, 4)
    output = model(torch.zeros(2, 8), torch.zeros(2, 4))
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    mask = torch.zeros_like(target, dtype=torch.bool)
    loss = masked_membrane_loss(output, torch.tensor([0.8, 0.2]), torch.ones(2, dtype=torch.bool), target, mask)
    loss.backward()
    assert all(head.weight.grad is None for head in model.auxiliary)


def test_interaction_aggregation_is_sparse_top_five_plus_noisy_or():
    frame = pd.DataFrame({
        "peptide_id": ["p"] * 6,
        "target_id": list("abcdef"),
        "s_aff_pair": [0.9, 0.8, 0.7, 0.6, 0.5, 0.1],
    })
    result = aggregate_pair_frame(frame, attention_noisy_or_weight=0.5).iloc[0]
    assert 0.0 < result["attention_top5"] < 1.0
    assert result["S_aff"] == pytest.approx(0.5 * result["attention_top5"] + 0.5 * result["noisy_or"])
    with pytest.raises(ValueError): aggregate_pair_frame(frame, top_k=4, attention_noisy_or_weight=0.5)


def test_numpy_and_torch_uncertainty_adjustment_match():
    mean = np.array([0.8, 0.4])
    std = np.array([0.1, 0.2])
    numpy_result = integrate_numpy(mean, mean, mean, GateParams(), std, std, std, 1.0)
    torch_result = GatedIntegrator()(torch.tensor(mean), torch.tensor(mean), torch.tensor(mean), torch.tensor(std), torch.tensor(std), torch.tensor(std))
    assert np.allclose(numpy_result["p_cons"], torch_result["p_cons"].numpy())


def test_main_entrypoints_do_not_import_legacy_experts():
    root = Path(__file__).parents[1]
    for relative in ("src/train/pretrain_public.py", "src/train/train_target_affinity.py", "src/score/score_candidates.py", "src/score/predict_target_affinity.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "PairAffinityExpert" not in source.replace("ProjectedPairAffinityExpert", "")
        assert "AMPPriorExpert" not in source.replace("PaperAMPPriorExpert", "")
        assert "MembraneMechanismExpert" not in source


def test_genetic_optimizer_exposes_exact_paper_parameters():
    config = GAConfig(population_size=12, generations=2)
    assert PAPER_SEEDS == (101, 211, 307, 401, 503)
    assert (config.tournament_size, config.elitism_fraction, config.crossover_probability, config.mutation_probability) == (3, 0.10, 0.60, 0.80)
    parent = "KALWQKALWQKALWQKALWQ"
    fitness = lambda seq: FitnessComponents(1.0 - abs(len(seq) - len(parent)) / 10, 0.5, 0.5)
    sequence, score = optimize(parent, fitness, seed=101, config=config)
    assert isinstance(sequence, str) and 0.0 <= score <= 1.0


def test_selection_rejects_non_200_pool_and_reserve_is_consumed_in_order():
    frame = pd.DataFrame({"sequence": ["KALWQKALWQ"], "R_t": [0.9], "selection_category": ["exploitation"]})
    with pytest.raises(ValueError, match="exactly 200"):
        freeze_round_selection(frame, np.zeros((1, 2)), budget=1, reserve_size=0, quotas={"exploitation": 1.0})
    reserve = pd.DataFrame({"reserve_order": [2, 1, 3], "sequence": ["B", "A", "C"]})
    assert next_reserve_candidate(reserve, set())["sequence"] == "A"
    assert next_reserve_candidate(reserve, {1})["sequence"] == "B"
