"""End-to-end pipeline test against a synthetic OptionDataSource (no IBKR).

Each mock name has a known spot, carry and vol; chains are generated as AMERICAN
prices via BAW so the pipeline must de-Americanize, imply the forward, build the
strip, and constant-maturity interpolate. Validates the orchestration and that
recovered L^Q is close to the lognormal closed form.
"""
from datetime import date, timedelta

import numpy as np
import pytest

from erp.config import Config
from erp.core.american import baw_price
from erp.data.base import OptionChain
from erp.pipeline.erp_snapshot import long_short, process_name, run_snapshot

ASOF = date(2026, 6, 26)


class MockSource:
    """Synthetic American chains at a per-symbol (vol, carry)."""

    def __init__(self, specs):
        self.specs = specs  # symbol -> dict(spot, sigma, q, drift)

    def get_underlying(self, symbol, asof):
        return self.specs[symbol]["spot"]

    def get_expiries(self, symbol, asof, around_days):
        return [asof + timedelta(days=d) for d in (21, 49)]  # brackets 30

    def get_risk_free(self, asof, tenor_days):
        return 0.045

    def get_chain(self, symbol, asof, expiry):
        sp = self.specs[symbol]
        S, sigma, q = sp["spot"], sp["sigma"], sp["q"]
        r = 0.045
        b = r - q
        tau = (expiry - asof).days / 365.0
        F = S * np.exp(b * tau)
        strikes = np.round(np.linspace(0.7 * F, 1.3 * F, 25), 1)
        cb, ca, pb, pa = [], [], [], []
        for K in strikes:
            c = baw_price(S, K, r, b, tau, sigma, True)
            p = baw_price(S, K, r, b, tau, sigma, False)
            cb.append(c * 0.99); ca.append(c * 1.01)
            pb.append(p * 0.99); pa.append(p * 1.01)
        return OptionChain(
            symbol=symbol, asof=asof, expiry=expiry, tau=tau, spot=S,
            strikes=strikes, call_bid=np.array(cb), call_ask=np.array(ca),
            put_bid=np.array(pb), put_ask=np.array(pa),
        )

    def get_total_return_history(self, symbol, asof, lookback_days):
        # deterministic mild-trend series; just needs to be positive and varied
        sp = self.specs[symbol]
        n = 260
        t = np.arange(n)
        drift = sp.get("drift", 0.0)
        wiggle = 1 + 0.01 * np.sin(t / 5.0) * sp["sigma"] * 10
        return sp["spot"] * np.exp(drift * t / 252) * wiggle


def test_single_name_recovers_lognormal_lq():
    src = MockSource({"AAA": {"spot": 100.0, "sigma": 0.25, "q": 0.0}})
    cfg = Config(universe=["AAA"])
    row = process_name(src, "AAA", ASOF, cfg)
    assert np.isfinite(row["L_Q"]), row["note"]
    # forward implied from parity should sit near S*exp(r*tau) (q=0)
    assert row["forward"] == pytest.approx(100 * np.exp(0.045 * 30 / 365), rel=2e-2)
    # L^Q ~ 0.5*sigma^2*tau (+ tiny rate terms) at the 30d constant maturity
    assert row["L_Q"] == pytest.approx(0.5 * 0.25**2 * 30 / 365, rel=0.1)
    assert row["fwd_disp"] < 0.05  # forward stable across ATM strikes


def test_forward_above_spot_with_dividends():
    """Positive q -> forward below spot; pipeline must not use spot as split."""
    src = MockSource({"DIV": {"spot": 80.0, "sigma": 0.3, "q": 0.06}})
    row = process_name(src, "DIV", ASOF, Config(universe=["DIV"]))
    assert np.isfinite(row["L_Q"]), row["note"]
    assert row["forward"] < 80.0  # carry q>r pulls forward under spot


def test_cross_section_sort_and_long_short():
    rng_specs = {
        f"S{i:02d}": {"spot": 50 + i, "sigma": 0.15 + 0.01 * i,
                      "q": 0.01, "drift": 0.02 * ((i % 5) - 2)}
        for i in range(12)
    }
    src = MockSource(rng_specs)
    cfg = Config(universe=list(rng_specs))
    df = run_snapshot(src, ASOF, cfg)
    assert df["erp"].notna().sum() == 12
    assert set(df["quartile"].dropna().unique()) == {1, 2, 3, 4}
    ls = long_short(df)
    assert ls["spread"] == ls["p4_mean_erp"] - ls["p1_mean_erp"]
    assert ls["p4_mean_erp"] >= ls["p1_mean_erp"]  # P4 is the high-ERP bucket
