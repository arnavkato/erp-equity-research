"""ERP cross-sectional backtest (spec Section 6, build-order step 7).

Monthly: compute each name's ERP at the rebalance date (point-in-time options),
sort into quartiles, form the long-short P4 - P1, and hold to the next rebalance.
No look-ahead: ERP at close(t) sorts the portfolio; the return is measured
forward from t to t+1. Reports quartile returns, the long-short mean with
Newey-West t-stats, and an equity curve.

Reuses pipeline.erp_snapshot.process_name verbatim — the same validated ERP code
that runs live. Only the data source (ThetaData) and the forward-return / stats
layer differ.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import Config
from ..core.constant_maturity import bracket_expiries
from ..stats.newey_west import nw_mean_tstat
from .erp_snapshot import process_name


def _prewarm_name(src, symbol: str, asof: date, cfg: Config):
    """Populate the cache with everything process_name will read for (symbol,
    asof) — but do NO de-Americalization. Runs serially in the connected main
    process (network-bound, fast) so the parallel workers below only ever hit a
    warm cache and never need the Terminal."""
    try:
        src.get_underlying(symbol, asof)
        expiries = src.get_expiries(symbol, asof, cfg.target_tau_days)
        exp_days = {e: (e - asof).days for e in expiries}
        lo_d, hi_d = bracket_expiries(list(exp_days.values()), cfg.target_tau_days)
        chosen = {d for d in (lo_d, hi_d) if d is not None}
        for exp, d in exp_days.items():
            if d in chosen:
                src.get_risk_free(asof, d)
                src.get_chain(symbol, asof, exp)
        src.get_total_return_history(symbol, asof, cfg.physical_fetch_days)
    except Exception:
        pass  # a name that fails to prewarm just becomes a FAIL row downstream


# Worker globals so each process builds its offline source once (not per task).
_WORKER_SRC = None
_WORKER_CFG = None


def _worker_init(cfg: Config):
    global _WORKER_SRC, _WORKER_CFG
    from ..data.thetadata import ThetaDataSource

    _WORKER_SRC = ThetaDataSource(cache=True)  # offline: never connects the Terminal
    _WORKER_CFG = cfg


def _worker(task):
    symbol, asof = task
    row = process_name(_WORKER_SRC, symbol, asof, _WORKER_CFG)
    row["date"] = asof
    return row


def monthly_rebalance_dates(start: date, end: date) -> list[date]:
    """First NYSE trading day of each month in [start, end]."""
    import pandas_market_calendars as mcal

    sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    days = [d.date() for d in pd.DatetimeIndex(sched.index)]
    firsts = {}
    for d in days:
        firsts.setdefault((d.year, d.month), d)
    return sorted(firsts.values())


def _price_at(ps: pd.Series, d: date) -> float:
    """Most recent close at or before date d."""
    if ps is None or ps.empty:
        return np.nan
    s = ps[ps.index <= d]
    return float(s.iloc[-1]) if len(s) else np.nan


def run_backtest(src, start: date, end: date, cfg: Config | None = None, log=print) -> dict:
    cfg = cfg or Config()
    rebals = monthly_rebalance_dates(start, end)
    log(f"rebalances: {len(rebals)} months {rebals[0]} .. {rebals[-1]}")

    # one price series per name for the whole window (cached) -> forward returns
    panels = {
        s: src.close_series(s, start - timedelta(days=15), end + timedelta(days=10))
        for s in cfg.universe
    }

    sort_dates = rebals[:-1]  # each holds to the next rebalance; last has no fwd
    tasks = [(s, t) for t in sort_dates for s in cfg.universe]

    # Phase 1: prewarm the cache serially in the connected process (network only,
    # no de-Am) so the workers below never need the Terminal.
    log(f"prewarming {len(tasks)} option chains ...")
    for j, t in enumerate(sort_dates, 1):
        for s in cfg.universe:
            _prewarm_name(src, s, t, cfg)
        log(f"  prewarmed {t} ({j}/{len(sort_dates)})")

    # Phase 2: de-Americalize + compute ERP in parallel across cores (workers read
    # the warm cache, never connect). This is the expensive, CPU-bound part.
    n_workers = max(1, (os.cpu_count() or 2) - 2)
    log(f"computing ERP for {len(tasks)} (name,date) pairs on {n_workers} workers ...")
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init,
                             initargs=(cfg,)) as ex:
        all_rows = pd.DataFrame(list(ex.map(_worker, tasks, chunksize=2)))

    # Attach the forward (hold-to-next-rebalance) return to every name/date, so
    # the per-name panel can be re-sorted on alternative signals without
    # recomputing any ERP.
    nextd = {rebals[i]: rebals[i + 1] for i in range(len(rebals) - 1)}

    def _fwd(symbol, t):
        t1 = nextd.get(t)
        if t1 is None:
            return np.nan
        c0, c1 = _price_at(panels[symbol], t), _price_at(panels[symbol], t1)
        return c1 / c0 - 1 if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 else np.nan

    all_rows["fwd"] = [_fwd(s, t) for s, t in zip(all_rows["symbol"], all_rows["date"])]

    # Phase 3: per-rebalance cross-sectional sort + forward returns.
    records = []
    for i in range(len(rebals) - 1):
        t, t1 = rebals[i], rebals[i + 1]
        df = all_rows[all_rows["date"] == t]
        valid = df["erp"].notna()
        nval = int(valid.sum())
        if nval < 8:
            log(f"{t}: only {nval} valid names — skipping")
            continue
        df = df[valid].copy()
        df["q"] = pd.qcut(df["erp"], 4, labels=[1, 2, 3, 4]).astype(int)
        df["fwd"] = [
            (lambda c0, c1: c1 / c0 - 1 if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 else np.nan)(
                _price_at(panels[s], t), _price_at(panels[s], t1)
            )
            for s in df["symbol"]
        ]
        qret = df.groupby("q")["fwd"].mean()
        ls = qret.get(4, np.nan) - qret.get(1, np.nan)
        rec = {"date": t, "n": nval, "ls": ls}
        rec.update({f"q{k}": qret.get(k, np.nan) for k in (1, 2, 3, 4)})
        records.append(rec)
        log(f"{t}: n={nval}  P1={qret.get(1, np.nan):+.4f} P4={qret.get(4, np.nan):+.4f} "
            f"L-S={ls:+.4f}  long={list(df[df.q==4].symbol)}")

    res = pd.DataFrame(records)
    if res.empty:
        return {"results": res, "stats": {}, "summary": "no valid rebalances"}

    res["equity"] = (1 + res["ls"].fillna(0)).cumprod()
    stats = nw_mean_tstat(res["ls"].dropna().values, lags=5)
    ann = res["ls"].mean() * 12
    vol = res["ls"].std() * np.sqrt(12)
    summary = {
        "months": len(res),
        "ls_mean_monthly": res["ls"].mean(),
        "ls_ann_return": ann,
        "ls_ann_vol": vol,
        "sharpe": ann / vol if vol else np.nan,
        "nw_t": stats["t"],
        "nw_p": stats["p"],
        "hit_rate": float((res["ls"] > 0).mean()),
        "q_means": {k: res[f"q{k}"].mean() for k in (1, 2, 3, 4)},
    }
    return {"results": res, "stats": stats, "summary": summary, "panel": all_rows}


def plot_backtest(out: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, summ = out["results"], out["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(res["date"], res["equity"], marker="o")
    ax1.axhline(1.0, color="k", lw=0.7)
    ax1.set_title(f"ERP long-short P4-P1 equity\nNW t={summ['nw_t']:.2f}  "
                  f"ann={summ['ls_ann_return']:+.1%}  Sharpe={summ['sharpe']:.2f}")
    ax1.set_ylabel("growth of $1"); ax1.grid(alpha=0.3)
    for lab in ax1.get_xticklabels():
        lab.set_rotation(45); lab.set_ha("right")

    qm = summ["q_means"]
    ax2.bar([f"Q{k}" for k in (1, 2, 3, 4)], [qm[k] for k in (1, 2, 3, 4)],
            color=["#d62728", "#ff9896", "#aec7e8", "#1f77b4"])
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_title("Mean monthly forward return by ERP quartile\n(Q1=low ERP/short, Q4=high ERP/long)")
    ax2.set_ylabel("mean fwd return"); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110)
    return path
