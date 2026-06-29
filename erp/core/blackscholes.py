"""Black-Scholes (Black-76 forward form) pricing and implied vol.

We price European options off the *forward* F rather than spot, because the ERP
strip (eq. 2.26) is replicated around the forward and we infer F from put-call
parity (Section 3a). Black-76 is exactly Black-Scholes written in forward terms:

    call = D * [ F*N(d1) - K*N(d2) ]
    put  = D * [ K*N(-d2) - F*N(-d1) ]
    d1   = (log(F/K) + 0.5*sigma^2*tau) / (sigma*sqrt(tau)),  d2 = d1 - sigma*sqrt(tau)

where D = exp(-r*tau) is the discount factor (PV factor). Everything is
vectorized over strike.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

SQRT_EPS = 1e-12


def bs_price(F, K, r, tau, sigma, is_call):
    """Present-value (discounted) Black-76 option price. Vectorized over K/sigma."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    D = np.exp(-r * tau)
    vol_sqrt = np.maximum(sigma * np.sqrt(tau), SQRT_EPS)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(np.asarray(is_call), call, put)


def bs_vega(F, K, r, tau, sigma):
    """dPrice/dsigma (per unit vol, not per %)."""
    D = np.exp(-r * tau)
    vol_sqrt = np.maximum(sigma * np.sqrt(tau), SQRT_EPS)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / vol_sqrt
    return D * F * norm.pdf(d1) * np.sqrt(tau)


def implied_vol(price, F, K, r, tau, is_call, lo=1e-4, hi=5.0):
    """Invert Black-76 for a single option. Returns NaN if price is outside
    the no-arbitrage bounds (a sign of stale/crossed quotes)."""
    D = np.exp(-r * tau)
    intrinsic = D * (F - K) if is_call else D * (K - F)
    intrinsic = max(intrinsic, 0.0)
    upper = D * F if is_call else D * K  # price bound as sigma -> inf
    if not np.isfinite(price) or price <= intrinsic + 1e-10 or price >= upper:
        return np.nan

    def obj(sig):
        return float(bs_price(F, K, r, tau, sig, is_call)) - price

    try:
        return brentq(obj, lo, hi, maxiter=100, xtol=1e-8)
    except ValueError:
        return np.nan


def implied_vol_vec(prices, F, strikes, r, tau, is_call):
    """Vectorized wrapper over implied_vol; is_call may be scalar or array."""
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)
    is_call = np.broadcast_to(is_call, strikes.shape)
    out = np.empty_like(strikes)
    for i in range(strikes.size):
        out.flat[i] = implied_vol(
            float(prices.flat[i]), F, float(strikes.flat[i]), r, tau, bool(is_call.flat[i])
        )
    return out
