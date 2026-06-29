"""Fama-MacBeth cross-sectional regression (spec Section 6, the paper's stat).

Each month, regress next-period returns on the signal across all names; the
FMB estimate is the time-series mean of the monthly slopes, with a Newey-West
t-stat. Standardizing the signal cross-sectionally each month makes the slope
read as "mean return per 1 cross-sectional SD of the signal" and removes scale
effects. This uses every name every month — far more efficient than collapsing
to quartile portfolios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .newey_west import nw_mean_tstat


def fama_macbeth(panel: pd.DataFrame, signal_cols, ret_col="fwd",
                 date_col="date", min_names=8, standardize=True) -> dict:
    """Univariate or multivariate FMB. signal_cols: str or list of str."""
    if isinstance(signal_cols, str):
        signal_cols = [signal_cols]
    rows = []
    for _, d in panel.groupby(date_col):
        d = d.dropna(subset=list(signal_cols) + [ret_col])
        if len(d) < min_names:
            continue
        X = []
        for c in signal_cols:
            x = d[c].to_numpy(float)
            if standardize:
                sd = x.std()
                if sd == 0:
                    x = x - x.mean()
                else:
                    x = (x - x.mean()) / sd
            X.append(x)
        X = np.column_stack([np.ones(len(d))] + X)
        beta, *_ = np.linalg.lstsq(X, d[ret_col].to_numpy(float), rcond=None)
        rows.append(beta[1:])  # drop intercept
    if not rows:
        return {"n_months": 0}
    B = np.array(rows)
    out = {"n_months": len(B)}
    for i, c in enumerate(signal_cols):
        st = nw_mean_tstat(B[:, i], lags=5)
        out[c] = {"gamma": st["mean"], "t": st["t"], "p": st["p"]}
    return out
