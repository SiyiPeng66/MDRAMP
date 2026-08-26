"""Runtime checks preventing same-round label leakage."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class PrequentialLeakageError(ValueError):
    """Raised when data visible to a state contains forbidden outcomes."""


def assert_no_current_round_labels(
    frame: pd.DataFrame,
    *,
    current_round: str,
    label_columns: Iterable[str],
    round_column: str = "round",
) -> None:
    """Reject non-null labels belonging to the round currently being scored.

    The check is intentionally strict: a current-round row with any populated
    outcome field is a leakage error, even when that field is not used by a
    particular model head.
    """

    if round_column not in frame.columns:
        return
    labels = [column for column in label_columns if column in frame.columns]
    if not labels:
        return
    current = frame[frame[round_column].astype(str) == str(current_round)]
    if current.empty:
        return
    populated = current[labels].notna().any(axis=1)
    if populated.any():
        ids = current.index[populated].tolist()[:10]
        raise PrequentialLeakageError(
            f"Current-round labels are visible for round {current_round!r}; "
            f"rows={ids}, columns={labels}."
        )


def assert_round_cutoff(
    frame: pd.DataFrame,
    *,
    visible_round: str,
    round_column: str = "round",
) -> None:
    """Reject rows from rounds later than the state visibility cutoff."""

    if round_column not in frame.columns:
        return
    observed = frame[round_column].dropna().astype(str)
    allowed = {f"R{i}" for i in range(int(str(visible_round).lstrip("R")) + 1)}
    future = sorted(set(observed) - allowed)
    if future:
        raise PrequentialLeakageError(
            f"Rows from future rounds {future} are visible to state {visible_round!r}."
        )
