"""Tests for parity forward (3a) and BAW de-Americanization (3b)."""
import numpy as np
import pytest

from erp.core.american import baw_price, de_americanize, european_price
from erp.core.blackscholes import bs_price
from erp.core.forward import implied_forward


# ---- parity forward ----------------------------------------------------------

def test_implied_forward_recovers_F():
    """Build European calls/puts off a known forward; parity must recover it."""
    F, r, tau, sigma = 105.0, 0.04, 30 / 365, 0.2
    strikes = np.linspace(90, 120, 31)
    calls = bs_price(F, strikes, r, tau, sigma, True)
    puts = bs_price(F, strikes, r, tau, sigma, False)
    F_hat, disp = implied_forward(strikes, calls, puts, r, tau)
    assert F_hat == pytest.approx(F, rel=1e-6)
    assert disp < 1e-6  # flat/consistent chain => tiny dispersion


def test_forward_differs_from_spot_with_dividends():
    """Sanity: with carry, the forward sits away from spot (why spot != split)."""
    S, r, q, tau = 100.0, 0.05, 0.03, 0.5
    F = S * np.exp((r - q) * tau)
    assert abs(F - S) > 0.5  # ~1 point away — enough to misplace the boundary


# ---- BAW de-Americanization --------------------------------------------------

def test_american_call_no_div_equals_european():
    """With q=0 (b=r) an American call should not be exercised early."""
    S, X, r, tau, sigma = 100.0, 100.0, 0.05, 0.5, 0.25
    amer = baw_price(S, X, r, r, tau, sigma, True)
    euro = european_price(S, X, r, r, tau, sigma, True)
    assert amer == pytest.approx(euro, rel=1e-6)


def test_american_put_has_early_exercise_premium():
    """American put > European put (early exercise has value)."""
    S, X, r, tau, sigma = 100.0, 100.0, 0.06, 1.0, 0.3
    b = r  # q = 0
    amer = baw_price(S, X, r, b, tau, sigma, False)
    euro = european_price(S, X, r, b, tau, sigma, False)
    assert amer > euro


def test_de_americanize_roundtrip_recovers_european():
    """Price an American option at vol sigma, de-Americanize, and we should get
    back the European price at that same sigma (the early-ex premium removed)."""
    S, X, r, q, tau, sigma = 100.0, 95.0, 0.05, 0.03, 0.5, 0.28
    b = r - q
    for is_call in (True, False):
        amer = baw_price(S, X, r, b, tau, sigma, is_call)
        euro = european_price(S, X, r, b, tau, sigma, is_call)
        recovered = de_americanize(amer, S, X, r, b, tau, is_call)
        assert recovered == pytest.approx(euro, rel=1e-4), is_call
        assert recovered <= amer + 1e-9  # European <= American
