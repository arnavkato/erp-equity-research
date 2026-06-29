"""Verify the entropy CALCULATION itself (not the option strip).

Two independent checks that L[R] = log E[R] - E[log R] and its cumulant estimator
are correct:

1. Lognormal closed form: if log R ~ N(m, v) then L[R] = v/2 exactly. Both the
   direct estimator and the cumulant estimator must recover v/2.
2. Cumulant identity on a NON-Gaussian return: the direct L[R] over horizon
   samples must equal the cumulant sum Σκ_n/n!, and adding higher orders must
   monotonically close the gap (o4 better than o2) when skew/kurtosis are present.
"""
import numpy as np

from erp.core.entropy import physical_entropy, physical_entropy_cumulant


def test_lognormal_closed_form():
    rng = np.random.default_rng(0)
    v = 0.02  # = sigma^2 * tau
    logR = rng.normal(-0.5 * v, np.sqrt(v), size=4_000_000)
    R = np.exp(logR)
    # direct: log E[R] - E[log R] should be v/2
    assert physical_entropy(R) == __import__("pytest").approx(v / 2, rel=0.03)


def test_cumulant_matches_direct_gaussian():
    rng = np.random.default_rng(1)
    h, s = 21, 0.012  # daily vol
    daily = rng.normal(0.0003, s, size=600_000)
    # horizon log return = sum of h iid daily; entropy ~ h*s^2/2
    lp_cum = physical_entropy_cumulant(daily, h, max_order=2)
    assert lp_cum == __import__("pytest").approx(h * s**2 / 2, rel=0.02)


def test_cumulant_identity_nongaussian():
    """Skewed, fat-tailed daily returns: cumulant sum must converge to the direct
    horizon entropy, and order 4 must beat order 2."""
    import pytest

    rng = np.random.default_rng(2)
    h = 21
    n_paths = 4_000_000
    # strongly skewed/fat-tailed daily: small diffusion + rare large down-jumps,
    # so skewness and kurtosis are large enough that order 2 is visibly biased.
    diff = rng.normal(0.0003, 0.008, size=(n_paths, h))
    jump = np.where(rng.random((n_paths, h)) < 0.02, -0.05, 0.0)  # 2%/day crash of -5%
    daily = diff + jump
    X = daily.sum(axis=1)            # horizon log return
    direct = physical_entropy(np.exp(X))

    flat_daily = daily.reshape(-1)
    o2 = physical_entropy_cumulant(flat_daily, h, max_order=2)
    o4 = physical_entropy_cumulant(flat_daily, h, max_order=4)

    # higher cumulants now matter: order 4 closes most of the order-2 gap and
    # lands within a few % of the direct horizon entropy.
    assert o4 == pytest.approx(direct, rel=0.05)
    assert abs(o4 - direct) < abs(o2 - direct)
