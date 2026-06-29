"""BKM risk-neutral moments on a synthetic flat-vol BS chain.

For a lognormal forward return y = ln(S_tau/F) ~ N(-v/2, v) with v = sigma^2*tau:
risk-neutral VAR = v, SKEW = 0, KURT = 3. The BKM strip must recover these.
Also checks realized_moments recovers the same on simulated Gaussian daily returns.
"""
import numpy as np
import pytest

from erp.core.bkm import bkm_risk_neutral_moments
from erp.core.blackscholes import bs_price
from erp.core.entropy import realized_moments
from erp.core.smile import OTMQuotes


def _flat_chain(F=100.0, r=0.03, tau=30 / 365, sigma=0.25, n=41, width=5.0):
    fsd = F * sigma * np.sqrt(tau)
    K = np.linspace(F - width * fsd, F + width * fsd, n)
    K = K[K > 0]
    is_call = K > F
    return OTMQuotes(F=F, r=r, tau=tau, strikes=K,
                    prices=bs_price(F, K, r, tau, sigma, is_call), is_call=is_call)


@pytest.mark.parametrize("sigma", [0.15, 0.30])
@pytest.mark.parametrize("tau", [30 / 365, 60 / 365])
def test_bkm_recovers_lognormal(sigma, tau):
    q = _flat_chain(sigma=sigma, tau=tau)
    m = bkm_risk_neutral_moments(q, n_std=10, n_grid=4000)
    assert m["var"] == pytest.approx(sigma**2 * tau, rel=2e-3)
    assert abs(m["skew"]) < 0.02      # lognormal forward return is ~symmetric
    assert m["kurt"] == pytest.approx(3.0, abs=0.05)


def test_realized_moments_gaussian():
    rng = np.random.default_rng(7)
    h, s = 21, 0.011
    daily = rng.normal(0, s, size=500_000)
    m = realized_moments(daily, h)
    assert m["var"] == pytest.approx(h * s**2, rel=0.02)
    assert abs(m["skew"]) < 0.05
    assert m["kurt"] == pytest.approx(3.0, abs=0.1)
