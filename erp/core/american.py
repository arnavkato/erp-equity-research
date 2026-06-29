"""Barone-Adesi-Whaley (1987) American option pricing + de-Americanization.

Single-name and ETF options are American; strict put-call parity fails and the
eq. 2.26 spanning assumes European prices. So we convert American -> European
BEFORE implying the forward or building the strip (spec Section 3b, pitfall 3).

De-Americanization recipe (standard practice):
  1. From the observed AMERICAN mid, imply a vol by inverting the BAW model.
  2. Re-price a EUROPEAN option (Black-Scholes-Merton) at that same vol.
The European-equivalent price strips out the early-exercise premium while
preserving the market's vol view.

Carry note: BAW needs a cost-of-carry b = r - q. We avoid *committing* to a
dividend yield by taking b as an explicit argument; the pipeline iterates it
(guess forward -> back out q -> re-de-Americanize). A modest b error only
perturbs the (already small, short-dated) early-exercise premium.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

SQRT_EPS = 1e-12


def _solve_bracketed(f, lo, hi, expand_up=True):
    """brentq with a bracket that auto-expands until the sign changes.

    At extreme vols the critical exercise price can move far outside a fixed
    bracket; expand toward the open end rather than throwing. Returns None if no
    sign change is found within a sane number of doublings (early exercise then
    effectively unreachable -> caller falls back to European)."""
    flo = f(lo)
    for _ in range(60):
        fhi = f(hi)
        if np.sign(flo) != np.sign(fhi):
            return brentq(f, lo, hi, maxiter=200)
        if expand_up:
            hi *= 1.5
        else:
            hi *= 0.5
            if hi < 1e-9:
                return None
    return None


def _bsm(S, X, r, b, tau, sigma, is_call):
    """European Black-Scholes-Merton with cost-of-carry b (= r - q)."""
    vol_sqrt = max(sigma * np.sqrt(tau), SQRT_EPS)
    d1 = (np.log(S / X) + (b + 0.5 * sigma**2) * tau) / vol_sqrt
    d2 = d1 - vol_sqrt
    carry = np.exp((b - r) * tau)  # = exp(-q*tau)
    if is_call:
        return S * carry * norm.cdf(d1) - X * np.exp(-r * tau) * norm.cdf(d2)
    return X * np.exp(-r * tau) * norm.cdf(-d2) - S * carry * norm.cdf(-d1)


def european_price(S, X, r, b, tau, sigma, is_call):
    """Public alias: European PV price in spot form."""
    return float(_bsm(S, X, r, b, tau, sigma, is_call))


def baw_price(S, X, r, b, tau, sigma, is_call):
    """American option price via Barone-Adesi-Whaley quadratic approximation."""
    sigma = max(float(sigma), 1e-6)
    eu = _bsm(S, X, r, b, tau, sigma, is_call)
    sig2 = sigma**2
    M = 2.0 * r / sig2
    N = 2.0 * b / sig2
    K = 1.0 - np.exp(-r * tau)
    if K <= 0:
        return float(eu)

    def d1(s):
        return (np.log(s / X) + (b + 0.5 * sig2) * tau) / (sigma * np.sqrt(tau))

    if is_call:
        if b >= r:  # no dividends -> never exercise an American call early
            return float(eu)
        q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2.0

        def f(s):  # value-matching: S* - X = c(S*) + (S*/q2)(1 - e^{(b-r)T}N(d1))
            return (
                s - X - _bsm(s, X, r, b, tau, sigma, True)
                - (s / q2) * (1 - np.exp((b - r) * tau) * norm.cdf(d1(s)))
            )

        s_star = _solve_bracketed(f, X, X * 100.0, expand_up=True)
        if s_star is None:
            return float(eu)
        A2 = (s_star / q2) * (1 - np.exp((b - r) * tau) * norm.cdf(d1(s_star)))
        if S < s_star:
            return float(eu + A2 * (S / s_star) ** q2)
        return float(S - X)
    else:
        q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K)) / 2.0

        def f(s):  # X - S** = p(S**) - (S**/q1)(1 - e^{(b-r)T}N(-d1))
            return (
                X - s - _bsm(s, X, r, b, tau, sigma, False)
                + (s / q1) * (1 - np.exp((b - r) * tau) * norm.cdf(-d1(s)))
            )

        s_star = _solve_bracketed(f, X, 1e-6, expand_up=False)
        if s_star is None:
            return float(eu)
        A1 = -(s_star / q1) * (1 - np.exp((b - r) * tau) * norm.cdf(-d1(s_star)))
        if S > s_star:
            return float(eu + A1 * (S / s_star) ** q1)
        return float(X - S)


def baw_implied_vol(price, S, X, r, b, tau, is_call, lo=1e-3, hi=5.0):
    """Invert BAW for the vol implied by an American option mid."""
    def obj(sig):
        return baw_price(S, X, r, b, tau, sig, is_call) - price

    try:
        return brentq(obj, lo, hi, maxiter=200, xtol=1e-8)
    except ValueError:
        return np.nan


def de_americanize(price, S, X, r, b, tau, is_call):
    """American mid -> European-equivalent PV price (same implied vol)."""
    iv = baw_implied_vol(price, S, X, r, b, tau, is_call)
    if not np.isfinite(iv):
        return np.nan
    return float(_bsm(S, X, r, b, tau, iv, is_call))
