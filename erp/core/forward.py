"""Parity-implied forward (spec Section 3a).

Do NOT use spot as the eq. 2.26 split point. The log contract is replicated
around the FORWARD, and F = S*exp((r-q)*tau) can sit meaningfully away from spot.
Infer F from put-call parity instead of estimating the dividend yield q:

    F = K + exp(r*tau) * ( C(K) - P(K) )

Derivation: parity C - P = S*exp(-q*tau) - K*exp(-r*tau); sub F = S*exp((r-q)*tau)
=> C - P = exp(-r*tau)(F - K). The dividend yield q cancels: option prices
already embed the market's dividend/carry. Use the strike closest to ATM
(tightest spread), or average the few nearest-ATM strikes.

IMPORTANT: this assumes EUROPEAN prices. For American single-name/ETF options,
de-Americanize first (core/american.py), then call this.
"""
from __future__ import annotations

import numpy as np


def implied_forward_at_strike(K, call, put, r, tau) -> float:
    """F from a single call/put pair at strike K via put-call parity."""
    return float(K + np.exp(r * tau) * (call - put))


def implied_forward(strikes, calls, puts, r, tau, n_atm=3):
    """Robust forward: average the parity forward across the n_atm strikes whose
    call/put prices are closest (|C - P| smallest => nearest ATM, tightest data).

    Returns (F, dispersion) where dispersion is the std across the strikes used —
    a stability check (spec Section 7: large dispersion => stale quotes or
    un-de-Americanized American contamination).
    """
    strikes = np.asarray(strikes, float)
    calls = np.asarray(calls, float)
    puts = np.asarray(puts, float)
    good = np.isfinite(calls) & np.isfinite(puts) & np.isfinite(strikes)
    strikes, calls, puts = strikes[good], calls[good], puts[good]
    if strikes.size == 0:
        return np.nan, np.nan

    # nearest-ATM = smallest |C - P| (where the parity line crosses zero)
    atm_order = np.argsort(np.abs(calls - puts))
    pick = atm_order[: min(n_atm, strikes.size)]
    fwds = strikes[pick] + np.exp(r * tau) * (calls[pick] - puts[pick])
    return float(np.mean(fwds)), float(np.std(fwds))
