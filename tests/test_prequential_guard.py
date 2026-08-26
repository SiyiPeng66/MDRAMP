import pandas as pd
import pytest

from src.loop.prequential_guard import (
    PrequentialLeakageError,
    assert_no_current_round_labels,
    assert_round_cutoff,
)


def test_current_round_labels_are_rejected():
    frame = pd.DataFrame(
        {"round": ["R1", "R1"], "mic": [None, 8.0], "sequence": ["AAA", "CCC"]}
    )
    with pytest.raises(PrequentialLeakageError):
        assert_no_current_round_labels(frame, current_round="R1", label_columns=["mic"])


def test_future_round_rows_are_rejected():
    frame = pd.DataFrame({"round": ["R0", "R2"]})
    with pytest.raises(PrequentialLeakageError):
        assert_round_cutoff(frame, visible_round="R1")


def test_unlabelled_current_round_is_allowed():
    frame = pd.DataFrame({"round": ["R1"], "mic": [None]})
    assert_no_current_round_labels(frame, current_round="R1", label_columns=["mic"])
