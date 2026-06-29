"""Live cross-sectional ERP snapshot over the Dow 30 (spec Section 6).

Per name: de-Americanize -> imply forward from parity -> filter -> build the OTM
strip -> L^Q at each bracketing expiry -> constant-maturity interpolate to the
target tau. Physical L^P from the trailing realized total-return distribution
over the matched horizon. ERP = L^P - L^Q. Then sort all names into quartiles and
form the long-short P4 - P1.

This module depends only on the OptionDataSource interface, so it runs against a
mock source in tests and against IBKR live with the same code path.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import Config
from ..core.american import de_americanize
from ..core.constant_maturity import bracket_expiries, interpolate
from ..core.bkm import bkm_risk_neutral_moments
from ..core.entropy import (
    physical_entropy,
    physical_entropy_cumulant,
    realized_moments,
    risk_neutral_entropy,
)
from ..core.forward import implied_forward
from ..core.smile import OTMQuotes
from ..data.base import OptionChain, OptionDataSource

TRADING_DAYS_PER_YEAR = 252


def _daily_log_returns(prices: np.ndarray) -> np.ndarray:
    """Daily log returns from a close series (oldest -> newest)."""
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    return np.diff(np.log(p)) if p.size > 2 else np.array([])


def realized_gross_returns(prices: np.ndarray, horizon_trading_days: int) -> np.ndarray:
    """Overlapping gross total returns R_t = P[t+h] / P[t] over the horizon."""
    p = np.asarray(prices, float)
    h = max(1, int(horizon_trading_days))
    if p.size <= h:
        return np.array([])
    return p[h:] / p[:-h]


def _european_otm_quotes(chain: OptionChain, r: float, cfg: Config):
    """De-Americanize the chain, imply the forward, and return the OTM strip
    (European PV mids split around F) plus diagnostics. Returns None if the
    filters reject the chain."""
    S, tau = chain.spot, chain.tau
    K = chain.strikes
    cmid, pmid = chain.call_mid(), chain.put_mid()

    # We only ever use OTM options (calls above F, puts below F) plus a near-ATM
    # overlap for the forward. De-Americanizing the deep-ITM side too would double
    # the per-strike BAW root-finds for nothing, so restrict to the OTM+ATM band
    # using spot as a cheap proxy for the split (F is within a few % of spot).
    want_call = K >= 0.95 * S
    want_put = K <= 1.05 * S

    # ---- de-Americanize with an iterated carry (spec 3b) --------------------
    b = r  # start from q = 0
    F = S
    for _ in range(cfg.carry_iters):
        eu_c = np.array([
            de_americanize(c, S, k, r, b, tau, True) if (w and np.isfinite(c)) else np.nan
            for c, k, w in zip(cmid, K, want_call)
        ])
        eu_p = np.array([
            de_americanize(p, S, k, r, b, tau, False) if (w and np.isfinite(p)) else np.nan
            for p, k, w in zip(pmid, K, want_put)
        ])
        F, disp = implied_forward(K, eu_c, eu_p, r, tau)
        if not np.isfinite(F) or F <= 0:
            return None
        # back out implied carry from the forward and re-de-Americanize
        b = r - np.log(F / S) / tau

    # ---- split OTM around F, apply filters (spec Section 5) -----------------
    min_price = cfg.min_price_ticks * cfg.min_tick
    is_call = K > F
    otm_price = np.where(is_call, eu_c, eu_p)
    keep = np.isfinite(otm_price) & (otm_price >= min_price) & (K > 0)
    n_call = int(np.sum(keep & is_call))
    n_put = int(np.sum(keep & ~is_call))
    if n_call < cfg.min_otm_per_side or n_put < cfg.min_otm_per_side:
        return None

    q = OTMQuotes(
        F=float(F), r=r, tau=tau,
        strikes=K[keep], prices=otm_price[keep], is_call=is_call[keep],
    )
    return q, float(F), float(disp)


def _lq_for_chain(chain: OptionChain, r: float, cfg: Config):
    built = _european_otm_quotes(chain, r, cfg)
    if built is None:
        return None
    q, F, disp = built
    lq = risk_neutral_entropy(q, n_std=cfg.n_std, n_grid=cfg.n_grid)
    bkm = bkm_risk_neutral_moments(q, n_std=cfg.n_std, n_grid=cfg.n_grid)
    return lq, F, disp, bkm


def process_name(source: OptionDataSource, symbol: str, asof: date, cfg: Config) -> dict:
    """Full per-name ERP. Returns a dict row (values NaN where the name fails)."""
    row = {"symbol": symbol, "L_Q": np.nan, "L_P": np.nan, "erp": np.nan,
           "forward": np.nan, "fwd_disp": np.nan, "n_expiries": 0,
           "vrp": np.nan, "srp": np.nan, "krp": np.nan,
           "rn_var": np.nan, "rn_skew": np.nan, "rn_kurt": np.nan,
           "phys_var": np.nan, "phys_skew": np.nan, "phys_kurt": np.nan, "note": ""}
    try:
        spot = source.get_underlying(symbol, asof)
        expiries = source.get_expiries(symbol, asof, cfg.target_tau_days)
    except Exception as e:  # noqa: BLE001 - a single bad name shouldn't kill the run
        row["note"] = f"data error: {e}"
        return row
    if not expiries:
        row["note"] = "no expiries"
        return row

    exp_days = {e: (e - asof).days for e in expiries}
    lo_d, hi_d = bracket_expiries(list(exp_days.values()), cfg.target_tau_days)
    chosen = sorted({d for d in (lo_d, hi_d) if d is not None})
    if not chosen:
        row["note"] = "no bracket"
        return row

    by_days = {}
    for exp, d in exp_days.items():
        if d in chosen and d not in by_days:
            by_days[d] = exp

    lq_by_days, rnm_by_days = {}, {}
    fwd, disp = np.nan, np.nan
    for d in chosen:
        exp = by_days[d]
        r = source.get_risk_free(asof, d)
        chain = source.get_chain(symbol, asof, exp)
        chain.spot = spot if not np.isfinite(chain.spot) else chain.spot
        res = _lq_for_chain(chain, r, cfg)
        if res is not None:
            lq_by_days[d], fwd, disp, rnm_by_days[d] = res

    if not lq_by_days:
        row["note"] = "no valid L_Q (filters/forward)"
        return row

    def _cm(key_fn):  # constant-maturity interpolate a per-expiry quantity
        return interpolate(cfg.target_tau_days,
                           lo_d, key_fn(lo_d), hi_d, key_fn(hi_d))

    lq = _cm(lambda d: lq_by_days.get(d))
    rn_var = _cm(lambda d: rnm_by_days.get(d, {}).get("var"))
    rn_skew = _cm(lambda d: rnm_by_days.get(d, {}).get("skew"))
    rn_kurt = _cm(lambda d: rnm_by_days.get(d, {}).get("kurt"))

    # ---- physical side: realized cumulants/moments over a 60-trading-day rolling
    # window, scaled to the option horizon (spec 3d/3e; paper's 60-day proxy) ---
    horizon_td = round(cfg.target_tau_days * TRADING_DAYS_PER_YEAR / 365)
    prices = source.get_total_return_history(symbol, asof, cfg.physical_fetch_days)
    dlr = _daily_log_returns(prices)[-cfg.physical_window_td:]
    lp = physical_entropy_cumulant(dlr, horizon_td, cfg.physical_cumulant_order)
    pm = realized_moments(dlr, horizon_td)

    # VRP/SRP/KRP = physical - risk-neutral (paper's convention, same as ERP)
    row.update(L_Q=lq, L_P=lp, erp=lp - lq, forward=fwd, fwd_disp=disp,
               n_expiries=len(lq_by_days),
               vrp=pm["var"] - rn_var, srp=pm["skew"] - rn_skew,
               krp=pm["kurt"] - rn_kurt,
               rn_var=rn_var, rn_skew=rn_skew, rn_kurt=rn_kurt,
               phys_var=pm["var"], phys_skew=pm["skew"], phys_kurt=pm["kurt"])
    return row


def run_snapshot(source: OptionDataSource, asof: date, cfg: Config | None = None) -> pd.DataFrame:
    """Compute ERP for the whole universe and attach quartile labels."""
    cfg = cfg or Config()
    rows = [process_name(source, s, asof, cfg) for s in cfg.universe]
    df = pd.DataFrame(rows)
    valid = df["erp"].notna()
    df["quartile"] = np.nan
    if valid.sum() >= 4:
        df.loc[valid, "quartile"] = (
            pd.qcut(df.loc[valid, "erp"], 4, labels=[1, 2, 3, 4]).astype("Int64")
        )
    return df.sort_values("erp", ascending=False, na_position="last").reset_index(drop=True)


def long_short(df: pd.DataFrame) -> dict:
    """P4 - P1 spread of the ERP sort (long high-ERP, short low-ERP)."""
    q = df.dropna(subset=["quartile"])
    if q.empty:
        return {"p4_mean_erp": np.nan, "p1_mean_erp": np.nan, "spread": np.nan}
    p4 = q.loc[q["quartile"] == 4, "erp"].mean()
    p1 = q.loc[q["quartile"] == 1, "erp"].mean()
    return {"p4_mean_erp": p4, "p1_mean_erp": p1, "spread": p4 - p1,
            "long": list(q.loc[q["quartile"] == 4, "symbol"]),
            "short": list(q.loc[q["quartile"] == 1, "symbol"])}
