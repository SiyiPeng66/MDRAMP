"""Activity-safety Pareto selection for R5."""

from __future__ import annotations

import pandas as pd


def pareto_front(frame: pd.DataFrame, *, activity: str = "activity_priority", risk: str = "hemolysis_risk") -> pd.DataFrame:
    required = {activity, risk}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing Pareto columns: {sorted(required - set(frame.columns))}")
    keep = []
    for i, row in frame.iterrows():
        dominated = ((frame[activity] >= row[activity]) & (frame[risk] <= row[risk]) & ((frame[activity] > row[activity]) | (frame[risk] < row[risk]))).any()
        if not dominated:
            keep.append(i)
    return frame.loc[keep].reset_index(drop=True)
