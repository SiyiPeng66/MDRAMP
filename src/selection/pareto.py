"""Measured panel-level activity–safety non-dominance (V6 S10.2)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pareto_front(frame, *, activity='activity_priority', risk='hemolysis_risk'):
    """Generic two-objective utility; not the measured lead-selection endpoint."""
    values = frame[[activity, risk]].to_numpy(dtype=float)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite Pareto values')
    transformed = values * np.array([1, -1])
    keep = [not ((transformed >= row).all(axis=1) & (transformed > row).any(axis=1)).any() for row in transformed]
    return frame.loc[keep].reset_index(drop=True)


def measured_pareto_front(frame, *, mic_columns, mhc5_column='MHC5'):
    """Compare every measured strain and MHC5; censoring must be resolved upstream.

    Use numeric bounds with optional <endpoint>_censored boolean columns.
    Retain candidates whenever dominance cannot be established from the bounds.
    """
    if not mic_columns: raise ValueError('At least one MIC column required')
    values = frame[list(mic_columns) + [mhc5_column]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Complete positive finite measured endpoints required')
    # Compare endpoint intervals conservatively. Right-censored values mean
    # (boundary, infinity); dominance must hold for every compatible value.
    lower, upper = values.copy(), values.copy()
    for j, column in enumerate(list(mic_columns) + [mhc5_column]):
        flag = column + '_censored'
        if flag in frame:
            text = frame[flag].astype(str).str.lower()
            if not text.isin(['true','false','0','1']).all(): raise ValueError('Invalid censor flags')
            upper[text.isin(['true','1']).to_numpy(), j] = np.inf
    # All objectives transformed to minimization; MHC5 is maximized.
    low, high = lower.copy(), upper.copy()
    low[:, -1], high[:, -1] = -upper[:, -1], -lower[:, -1]
    keep = []
    for i in range(len(frame)):
        certain = (high <= low[i]).all(axis=1) & (high < low[i]).any(axis=1)
        certain[i] = False
        keep.append(not certain.any())
    return frame.loc[keep].reset_index(drop=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--mic-columns',nargs='+',required=True); p.add_argument('--mhc5-column',default='MHC5')
    a=p.parse_args(); result=measured_pareto_front(pd.read_csv(a.input),mic_columns=a.mic_columns,mhc5_column=a.mhc5_column)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); result.to_csv(a.output,index=False)

if __name__=='__main__': main()
