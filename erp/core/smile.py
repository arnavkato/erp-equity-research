"""IV-space smile interpolation and dense-grid resampling (spec Section 5).

The whole point of this module: NEVER interpolate or extrapolate option prices
directly. The 1/K^2 weight in the entropy strip blows up small wing errors, so
we work in implied-vol space — cubic spline across strikes, flat extrapolation
beyond the observed range — then convert back to prices on a dense grid before
integrating. This is the same recipe behind the VIX and BKM moments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .blackscholes import bs_price, implied_vol


@dataclass
class OTMQuotes:
    """OTM option mids on a single expiry, already split around the forward.

    strikes/prices/is_call are aligned arrays. Convention: OTM means calls for
    K > F and puts for K < F. r and tau describe the discounting/horizon.
    """

    F: float
    r: float
    tau: float
    strikes: np.ndarray
    prices: np.ndarray  # present-value mids
    is_call: np.ndarray  # bool array


def quotes_to_iv(q: OTMQuotes):
    """Map each OTM mid to its Black-76 implied vol. Drops un-invertible quotes."""
    from .blackscholes import implied_vol_vec

    iv = implied_vol_vec(q.prices, q.F, q.strikes, q.r, q.tau, q.is_call)
    good = np.isfinite(iv) & (iv > 0)
    return q.strikes[good], iv[good]


def build_smile(strikes, ivs):
    """Cubic spline of IV vs strike with FLAT extrapolation past the wings.

    Returns a callable sigma(K). Requires >= 2 points; uses linear if only 2.
    """
    order = np.argsort(strikes)
    k = np.asarray(strikes, float)[order]
    v = np.asarray(ivs, float)[order]
    # de-duplicate strikes (can happen across call/put overlap near ATM)
    k, idx = np.unique(k, return_index=True)
    v = v[idx]
    if k.size < 2:
        raise ValueError("need >= 2 IV points to build a smile")

    lo_v, hi_v = v[0], v[-1]
    lo_k, hi_k = k[0], k[-1]
    if k.size >= 4:
        spline = CubicSpline(k, v, bc_type="natural")
    else:
        # too few points for a stable cubic; fall back to linear in strike
        def spline(x):
            return np.interp(x, k, v)

    def sigma(K):
        K = np.asarray(K, float)
        inside = spline(np.clip(K, lo_k, hi_k))
        # flat extrapolation: hold the edge vol constant beyond observed strikes
        out = np.where(K < lo_k, lo_v, np.where(K > hi_k, hi_v, inside))
        return out

    return sigma


def dense_strip(q: OTMQuotes, n_std=8.0, n_grid=2000):
    """Resample the smile onto a dense strike grid and return (K, otm_price).

    Grid spans the forward +/- n_std forward-standard-deviations (using an ATM
    vol estimate), which truncates the 1/K^2 tails at a documented bound
    (spec Section 5.4). otm_price(K) is a call for K>F, else a put.
    """
    ks, ivs = quotes_to_iv(q)
    if ks.size < 2:
        raise ValueError("not enough invertible OTM quotes for a smile")
    sigma = build_smile(ks, ivs)

    atm_vol = float(sigma(q.F))
    fsd = q.F * atm_vol * np.sqrt(q.tau)  # forward standard deviation
    k_lo = max(1e-6, q.F - n_std * fsd)
    k_hi = q.F + n_std * fsd
    grid = np.linspace(k_lo, k_hi, n_grid)

    vols = sigma(grid)
    is_call = grid > q.F
    prices = bs_price(q.F, grid, q.r, q.tau, vols, is_call)
    return grid, prices
