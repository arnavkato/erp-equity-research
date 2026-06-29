"""Newey-West (HAC) t-statistic of a mean (spec Section 6, stats).

The long-short return series is autocorrelated/heteroskedastic, so a plain t-stat
overstates significance. Regress the series on a constant with HAC standard
errors (5 lags, the paper's choice) — the constant's t-stat is the Newey-West
t-stat of the mean.
"""
from __future__ import annotations

import numpy as np


def nw_mean_tstat(x, lags: int = 5) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {"mean": np.nan, "t": np.nan, "p": np.nan, "n": int(x.size)}
    if x.size < 3:
        return out
    import statsmodels.api as sm

    res = sm.OLS(x, np.ones((x.size, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out.update(mean=float(res.params[0]), t=float(res.tvalues[0]), p=float(res.pvalues[0]))
    return out
