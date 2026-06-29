"""Compare cross-sectional signals on the cached ERP panel — no recompute.

The raw-ERP sort is dominated by each stock's variance LEVEL (corr ~0.76 with
L_Q), so it effectively ranks high-vol vs low-vol names. This tests normalizations
that strip the level out, per the idea "scale ERP by how much IV the stock has":

  raw            ERP = L_P - L_Q
  erp_over_lq    ERP / L_Q              ~ realized/implied variance ratio - 1
  var_ratio      L_P / L_Q             (= 1 + erp_over_lq; same sort)
  erp_z          per-name z-score of ERP over its own history (time-series demean)
  erp_resid      ERP minus its cross-sectional fit on L_Q each month (orthogonalize)

For each: form monthly quartile long-short P4-P1, report Newey-West t and the
quartile-mean monotonicity.  python -m erp.analyze_signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats.newey_west import nw_mean_tstat

PANEL = "erp/backtest_panel.csv"


def _add_signals(p: pd.DataFrame) -> pd.DataFrame:
    p = p.dropna(subset=["erp", "L_Q", "fwd"]).copy()
    p["raw"] = p["erp"]
    p["erp_over_lq"] = p["erp"] / p["L_Q"]
    p["var_ratio"] = p["L_P"] / p["L_Q"]
    # per-name time-series z-score (is this name's ERP high vs its own norm?)
    g = p.groupby("symbol")["erp"]
    p["erp_z"] = (p["erp"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    # cross-sectional residual of ERP on L_Q within each month (orthogonalize level)
    p["erp_resid"] = np.nan
    for _, idx in p.groupby("date").groups.items():
        d = p.loc[idx]
        if len(d) < 5:
            continue
        b1, b0 = np.polyfit(d["L_Q"].values, d["erp"].values, 1)
        p.loc[idx, "erp_resid"] = d["erp"].values - (b0 + b1 * d["L_Q"].values)
    return p


def _eval(p: pd.DataFrame, sig: str) -> dict:
    rows = []
    for t, d in p.groupby("date"):
        d = d.dropna(subset=[sig, "fwd"])
        if d["erp"].notna().sum() < 8 or d[sig].nunique() < 4:
            continue
        q = pd.qcut(d[sig], 4, labels=[1, 2, 3, 4], duplicates="drop")
        if q.nunique() < 4:
            continue
        qret = d.groupby(q, observed=True)["fwd"].mean()
        rows.append({"date": t, "ls": qret.get(4, np.nan) - qret.get(1, np.nan),
                     **{f"q{k}": qret.get(k, np.nan) for k in (1, 2, 3, 4)}})
    r = pd.DataFrame(rows)
    if r.empty:
        return {"signal": sig, "n": 0}
    st = nw_mean_tstat(r["ls"].dropna().values, lags=5)
    qm = [r[f"q{k}"].mean() for k in (1, 2, 3, 4)]
    mono = qm[0] < qm[1] < qm[2] < qm[3]  # strictly increasing Q1->Q4
    return {"signal": sig, "n": len(r), "ls_mean": r["ls"].mean(),
            "ann": r["ls"].mean() * 12, "nw_t": st["t"], "nw_p": st["p"],
            "hit": float((r["ls"] > 0).mean()),
            "q1..q4": "  ".join(f"{x:+.4f}" for x in qm), "monotonic": mono}


def main():
    p = _add_signals(pd.read_csv(PANEL))
    sigs = ["raw", "erp_over_lq", "var_ratio", "erp_z", "erp_resid"]
    res = pd.DataFrame([_eval(p, s) for s in sigs])
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
