"""Bakshi-Kapadia-Madan (2003) risk-neutral moments (spec Appendix B.1).

Same OTM-option strip as the entropy/VIX machinery, but with the BKM payoff
kernels for the quadratic/cubic/quartic contracts. Referenced to the forward F
(so y = ln(S_tau / F) is the forward log return, a Q-martingale exponent), which
makes the call and put kernels identical and keeps everything consistent with our
forward-based L^Q.

With x = ln(K/F) the kernels are uniform across OTM calls (K>F) and puts (K<F):
    V (quadratic): 2(1 - x) / K^2          -> E^Q[y^2]
    W (cubic):     (6x - 3x^2) / K^2        -> E^Q[y^3]
    X (quartic):   (12x^2 - 4x^3) / K^2      -> E^Q[y^4]
Each integral is taken to forward value (x R_f), matching the strip. Then
    mu   = -V/2 - W/6 - X/24                 (E^Q[y], from E^Q[e^y]=1)
    VAR  = V - mu^2
    SKEW = (W - 3 mu V + 2 mu^3) / VAR^1.5
    KURT = (X - 4 mu W + 6 mu^2 V - 3 mu^4) / VAR^2
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .smile import OTMQuotes, dense_strip


def bkm_risk_neutral_moments(q: OTMQuotes, **grid_kw) -> dict:
    grid, prices = dense_strip(q, **grid_kw)
    x = np.log(grid / q.F)
    R_f = np.exp(q.r * q.tau)
    inv_k2 = 1.0 / grid**2
    V = R_f * simpson(2.0 * (1.0 - x) * inv_k2 * prices, x=grid)
    W = R_f * simpson((6.0 * x - 3.0 * x**2) * inv_k2 * prices, x=grid)
    X = R_f * simpson((12.0 * x**2 - 4.0 * x**3) * inv_k2 * prices, x=grid)

    mu = -V / 2.0 - W / 6.0 - X / 24.0
    var = V - mu**2
    if not np.isfinite(var) or var <= 0:
        return {"var": np.nan, "skew": np.nan, "kurt": np.nan}
    skew = (W - 3.0 * mu * V + 2.0 * mu**3) / var**1.5
    kurt = (X - 4.0 * mu * W + 6.0 * mu**2 * V - 3.0 * mu**4) / var**2
    return {"var": float(var), "skew": float(skew), "kurt": float(kurt)}
