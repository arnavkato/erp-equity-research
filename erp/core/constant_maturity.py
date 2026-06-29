"""Constant-maturity interpolation (spec Section 3c).

Equity expiries are discrete, so a chain rarely sits exactly at the target
horizon. Compute L^Q for the two expiries bracketing the target tau and
interpolate linearly in days-to-maturity (the VIX constant-maturity trick).
Falls back to the single nearest expiry if only one is usable.
"""
from __future__ import annotations

import numpy as np


def bracket_expiries(expiries_days, target_days):
    """Pick the expiry just below and just above target. Returns (lo, hi) day
    counts; either may be None if the target is outside the listed range."""
    days = sorted(d for d in expiries_days if d is not None and d > 0)
    lo = max((d for d in days if d <= target_days), default=None)
    hi = min((d for d in days if d >= target_days), default=None)
    return lo, hi


def interpolate(target_days, lo_days, lo_val, hi_days, hi_val):
    """Linear-in-DTE interpolation of an entropy value to the target horizon."""
    if lo_days is None and hi_days is None:
        return np.nan
    if lo_days is None or lo_val is None or not np.isfinite(lo_val):
        return hi_val
    if hi_days is None or hi_val is None or not np.isfinite(hi_val):
        return lo_val
    if hi_days == lo_days:
        return 0.5 * (lo_val + hi_val)
    w = (target_days - lo_days) / (hi_days - lo_days)
    return (1 - w) * lo_val + w * hi_val
