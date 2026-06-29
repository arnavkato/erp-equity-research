"""Gate test (spec build-order step 1 / Section 7): on a flat-vol Black-Scholes
chain the entropy strip has a known closed form.

For a lognormal underlying with constant vol sigma over horizon tau, the
1/K^2 OTM-option strip equals the variance-swap value:

    R_f * int (1/K^2) O(K) dK  ==  sigma^2 * tau / 2

This is the same identity the VIX is built on. If this passes, the
smile -> dense-grid -> integrate machinery is sound. Everything downstream is
gated on it.
"""
import numpy as np
import pytest

from erp.core.blackscholes import bs_price
from erp.core.entropy import log_contract_strip, risk_neutral_entropy
from erp.core.smile import OTMQuotes


def make_flat_chain(F=100.0, r=0.04, tau=30 / 365, sigma=0.20, n=41, width=4.0):
    """Synthetic OTM chain at constant IV = sigma. Strikes span F +/- width fsd."""
    fsd = F * sigma * np.sqrt(tau)
    strikes = np.linspace(F - width * fsd, F + width * fsd, n)
    strikes = strikes[strikes > 0]
    is_call = strikes > F
    prices = bs_price(F, strikes, r, tau, sigma, is_call)
    return OTMQuotes(F=F, r=r, tau=tau, strikes=strikes, prices=prices, is_call=is_call)


@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35])
@pytest.mark.parametrize("tau", [30 / 365, 60 / 365])
def test_strip_equals_half_sigma2_tau(sigma, tau):
    q = make_flat_chain(sigma=sigma, tau=tau)
    strip = log_contract_strip(q, n_std=8.0, n_grid=4000)
    expected = 0.5 * sigma**2 * tau
    assert strip == pytest.approx(expected, rel=2e-3), (strip, expected)


def test_risk_neutral_entropy_closed_form():
    F, r, tau, sigma = 100.0, 0.04, 30 / 365, 0.22
    q = make_flat_chain(F=F, r=r, tau=tau, sigma=sigma)
    lq = risk_neutral_entropy(q, n_std=8.0, n_grid=4000)
    R_f = np.exp(r * tau)
    expected = np.log(R_f) - (R_f - 1.0) + 0.5 * sigma**2 * tau
    assert lq == pytest.approx(expected, rel=1e-3)
