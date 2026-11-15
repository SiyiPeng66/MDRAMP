import pandas as pd
import pytest
from src.train.train_target_affinity import validate_training_partition


@pytest.mark.parametrize('split', ['test', 'validation', None])
def test_interaction_rejects_nontraining_partition(split):
    with pytest.raises(ValueError, match='only train'):
        validate_training_partition(pd.DataFrame({'split': ['train', split]}))


def test_interaction_accepts_train_partition():
    validate_training_partition(pd.DataFrame({'split': ['train', 'train']}))
