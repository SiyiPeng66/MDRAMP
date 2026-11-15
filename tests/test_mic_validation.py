import numpy as np
import torch
from src.train.train_mic_head import train_mic_head
from src.models.mic_regressor import MICRegressionHead
from src.loop.prequential_guard import assert_round_cutoff, PrequentialLeakageError
import pandas as pd
import pytest


def test_missing_round_provenance_rejected():
    with pytest.raises(PrequentialLeakageError):
        assert_round_cutoff(pd.DataFrame({'mic_log2': [2.]}), visible_round='R1')


def test_validation_selected_head_is_finite():
    torch.manual_seed(11)
    model = MICRegressionHead(2, 1)
    arrays = (np.ones((4, 2), np.float32), np.zeros((4, 1), np.float32),
              np.ones((4, 3), np.float32), np.full(4, 3., np.float32), np.zeros(4, bool))
    losses = train_mic_head(model, *arrays, epochs=2, validation=arrays)
    assert len(losses) == 2 and np.isfinite(losses).all()
    assert all(torch.isfinite(p).all() for p in model.parameters())
