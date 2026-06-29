"""ThetaData historical option-data loader (spec Section 4, backtest source).

Implements the same OptionDataSource interface as IBKRSource, so the validated
pipeline runs unchanged on history. Unlike IBKR (live-only, deletes expired
options, and here lacked an OPRA bid/ask entitlement), ThetaData serves dense
historical EOD chains with real NBBO bid/ask + the SOFR curve.

The `thetadata` SDK auto-launches its Terminal (needs Java) from email/password;
there is no manual terminal step. Returned objects are pandas DataFrames.

A thin parquet cache (keyed by call) means re-runs of a backtest don't re-hit the
API — important since a multi-year monthly backtest pulls thousands of chains.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.base import OptionChain

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "thetadata"


class ThetaDataSource:
    def __init__(self, email: str | None = None, password: str | None = None,
                 cache: bool = True, rate_symbol: str = "SOFR"):
        self.email = email or os.getenv("THETA_EMAIL")
        self.password = password or os.getenv("THETA_PASSWORD")
        self.rate_symbol = rate_symbol
        self.cache = cache
        self._tc = None
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def connect(self):
        from thetadata import ThetaClient

        self._tc = ThetaClient(email=self.email, password=self.password,
                               dataframe_type="pandas")
        return self

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        # SDK manages the Terminal lifecycle; nothing to close explicitly.
        self._tc = None

    # -- parquet cache ---------------------------------------------------------
    def _cached(self, key: str, fn):
        if not self.cache:
            return fn()
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{h}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        if self._tc is None:
            # offline (e.g. a parallel worker that only reads the cache): a miss
            # can't be fetched — return None and let the caller degrade that name.
            return None
        df = fn()
        try:
            df.to_parquet(path)
        except Exception:
            pass  # never let a cache write failure break a run
        return df

    # -- OptionDataSource ------------------------------------------------------
    def get_underlying(self, symbol: str, asof: date) -> float:
        df = self._cached(
            f"stk_eod:{symbol}:{asof}",
            lambda: self._tc.stock_history_eod(symbol, asof, asof),
        )
        if df is None or df.empty:
            return np.nan
        row = df.iloc[-1]
        mid = _mid(row.get("bid"), row.get("ask"))
        return float(mid if np.isfinite(mid) else row["close"])

    def get_expiries(self, symbol: str, asof: date, around_days: int) -> list[date]:
        df = self._cached(
            f"exps:{symbol}",
            lambda: self._tc.option_list_expirations(symbol),
        )
        exps = pd.to_datetime(df["expiration"].astype(str)).dt.date
        lo, hi = around_days - 25, around_days + 35
        return sorted(e for e in exps if lo <= (e - asof).days <= hi)

    def get_chain(self, symbol: str, asof: date, expiry: date) -> OptionChain:
        df = self._cached(
            f"opt_eod:{symbol}:{asof}:{expiry}",
            lambda: self._tc.option_history_eod(asof, asof, symbol, expiry,
                                                strike="*", right="both"),
        )
        spot = self.get_underlying(symbol, asof)
        tau = (expiry - asof).days / 365.0
        if df is None or df.empty:
            empty = np.array([])
            return OptionChain(symbol, asof, expiry, tau, spot, empty,
                               empty, empty, empty, empty)

        # pivot to one row per strike with call/put bid/ask
        df = df.copy()
        df["right"] = df["right"].str.upper()
        strikes = np.array(sorted(df["strike"].unique()), float)
        idx = {k: i for i, k in enumerate(strikes)}
        n = strikes.size
        cb = np.full(n, np.nan); ca = np.full(n, np.nan)
        pb = np.full(n, np.nan); pa = np.full(n, np.nan)
        for _, r in df.iterrows():
            i = idx[float(r["strike"])]
            b, a = _clean(r["bid"]), _clean(r["ask"])
            # fall back to the EOD close when a side has no NBBO quote
            if not (np.isfinite(b) and np.isfinite(a)):
                c = _clean(r["close"])
                b = b if np.isfinite(b) else c
                a = a if np.isfinite(a) else c
            if r["right"] == "CALL":
                cb[i], ca[i] = b, a
            else:
                pb[i], pa[i] = b, a
        return OptionChain(symbol, asof, expiry, tau, spot, strikes, cb, ca, pb, pa)

    def get_risk_free(self, asof: date, tenor_days: int) -> float:
        df = self._cached(
            f"rate:{self.rate_symbol}:{asof}",
            lambda: self._tc.interest_rate_history_eod(self.rate_symbol, asof, asof),
        )
        if df is None or df.empty:
            return 0.045
        return float(df.iloc[-1]["rate"]) / 100.0  # ThetaData reports percent

    def _stock_eod_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """stock_history_eod, chunked into <=360-day windows (ThetaData caps a
        single stock-history request at 365 days) and concatenated. Each chunk is
        cached independently."""
        from datetime import timedelta

        frames = []
        cur = start
        while cur <= end:
            chunk_end = min(end, cur + timedelta(days=360))
            try:
                df = self._cached(
                    f"stk_eod_rng:{symbol}:{cur}:{chunk_end}",
                    lambda c=cur, e=chunk_end: self._tc.stock_history_eod(symbol, c, e),
                )
            except Exception:
                # e.g. deep history beyond the data-tier entitlement — skip the
                # chunk rather than crash; the caller degrades on short history.
                df = None
            if df is not None and not df.empty:
                frames.append(df)
            cur = chunk_end + timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def close_series(self, symbol: str, start: date, end: date) -> pd.Series:
        """Daily EOD close indexed by trading date — used to measure realized
        forward returns between rebalances in the backtest."""
        df = self._stock_eod_raw(symbol, start, end)
        if df.empty:
            return pd.Series(dtype=float)
        d = pd.to_datetime(df["created"]).dt.date
        s = pd.Series(df["close"].to_numpy(float), index=pd.Index(d, name="date"))
        return s[~s.index.duplicated(keep="last")].sort_index()

    def get_total_return_history(self, symbol: str, asof: date, lookback_days: int) -> np.ndarray:
        from datetime import timedelta

        df = self._stock_eod_raw(symbol, asof - timedelta(days=lookback_days), asof)
        if df.empty:
            return np.array([])
        # NOTE: raw close = price return (dividends omitted). Dow div yields are
        # small over a 30d horizon; refine with a corporate-actions adjust later.
        return df.sort_values("created")["close"].to_numpy(float)


def _clean(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) and x > 0 else np.nan


def _mid(b, a):
    b, a = _clean(b), _clean(a)
    if np.isfinite(b) and np.isfinite(a):
        return 0.5 * (b + a)
    return np.nan
