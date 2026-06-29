"""Entropy of returns under P and Q (Case 1, the log specification).

Risk-neutral entropy from options (eq. 2.26, derived in Appendix A.4 via the
Carr-Madan log-contract spanning):

    L^Q[R] = log(R_f) - (R_f - 1)
             + R_f * [ int_F^inf (1/K^2) C(K) dK + int_0^F (1/K^2) P(K) dK ]

C(K), P(K) are present-value mids; the leading R_f converts them to forward
value. SIGN: we use the Appendix A.4 minus sign on (R_f - 1) (the main text
prints +; the appendix is the actual derivation). Magnitude is tiny since
R_f ~ 1 at short horizons, but be correct.

Physical entropy (eq. 2.13) from realized gross returns:

    L^P[R] = log(E[R]) - E[log R]
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .smile import OTMQuotes, dense_strip


def log_contract_strip(q: OTMQuotes, **grid_kw) -> float:
    """The 1/K^2 OTM-option strip, returned in FORWARD value:

        R_f * [ int_F^inf C/K^2 dK + int_0^F P/K^2 dK ]

    For a flat-vol Black-Scholes chain this equals sigma^2 * tau / 2 (the
    variance-swap / VIX identity) — that's the synthetic unit-test gate.
    """
    grid, prices = dense_strip(q, **grid_kw)
    integrand = prices / grid**2
    R_f = np.exp(q.r * q.tau)
    return float(R_f * simpson(integrand, x=grid))


def risk_neutral_entropy(q: OTMQuotes, **grid_kw) -> float:
    """L^Q[R], Case 1 (eq. 2.26). Reused verbatim by both pipelines."""
    R_f = np.exp(q.r * q.tau)
    strip = log_contract_strip(q, **grid_kw)
    return float(np.log(R_f) - (R_f - 1.0) + strip)


def physical_entropy(gross_returns) -> float:
    """L^P[R] = log(E[R]) - E[log R] over a sample of realized gross returns R.

    Inputs must be GROSS total returns (price + dividends) over the SAME horizon
    tau as the option side (spec Section 3d/3e). Gross returns must be > 0.
    """
    R = np.asarray(gross_returns, dtype=float)
    R = R[np.isfinite(R) & (R > 0)]
    if R.size < 2:
        return np.nan
    return float(np.log(R.mean()) - np.log(R).mean())


def physical_entropy_cumulant(daily_log_returns, horizon_days: int, max_order: int = 4) -> float:
    """L^P over a horizon, estimated from DAILY log-return cumulants scaled to the
    horizon — far less noisy than a handful of overlapping horizon returns.

    Identity: L^P[R] = log E[R] - E[log R] = sum_{n>=2} kappa_n / n!, where
    kappa_n are the cumulants of the horizon log-return X = sum of daily log
    returns. Under iid daily returns, kappa_n(X) = h * kappa_n(daily). So:

        L^P ~= h*c2/2 + h*c3/6 + h*c4/24      (c_n = daily log-return cumulants)

    Uses all daily observations in the window (hundreds), stays horizon-matched,
    and reduces to (1/2)*sigma^2*tau in the Gaussian limit (the lognormal check).
    """
    x = np.asarray(daily_log_returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return np.nan
    h = max(1, int(horizon_days))
    mu = x.mean()
    xc = x - mu
    c2 = np.mean(xc**2)                       # variance (cumulant 2)
    val = h * c2 / 2.0
    if max_order >= 3:
        c3 = np.mean(xc**3)                   # cumulant 3
        val += h * c3 / 6.0
    if max_order >= 4:
        c4 = np.mean(xc**4) - 3.0 * c2**2     # excess (cumulant 4)
        val += h * c4 / 24.0
    return float(val)


def realized_moments(daily_log_returns, horizon_days: int) -> dict:
    """Physical variance/skewness/kurtosis of the horizon return, from daily
    realized cumulants scaled to the horizon (iid: kappa_n(tau) = h*kappa_n(daily)).
    Returns NON-excess kurtosis to match the BKM convention (Gaussian -> 3)."""
    x = np.asarray(daily_log_returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return {"var": np.nan, "skew": np.nan, "kurt": np.nan}
    h = max(1, int(horizon_days))
    xc = x - x.mean()
    c2 = np.mean(xc**2)
    c3 = np.mean(xc**3)
    c4 = np.mean(xc**4) - 3.0 * c2**2          # excess (cumulant 4)
    k2, k3, k4 = h * c2, h * c3, h * c4         # horizon cumulants
    if k2 <= 0:
        return {"var": np.nan, "skew": np.nan, "kurt": np.nan}
    return {"var": float(k2), "skew": float(k3 / k2**1.5),
            "kurt": float(3.0 + k4 / k2**2)}


def erp(q: OTMQuotes, gross_returns, **grid_kw) -> dict:
    """Entropy risk premium ERP = L^P - L^Q (eq. 2.23), plus its components."""
    lq = risk_neutral_entropy(q, **grid_kw)
    lp = physical_entropy(gross_returns)
    return {"erp": lp - lq, "L_P": lp, "L_Q": lq}
