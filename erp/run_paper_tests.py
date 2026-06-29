"""The paper's actual tests on the Dow 30, on the augmented panel.

Mirrors Chabi-Yo/Doshi/Zurita:
  Table 2  raw ERP quartile sort (long P4 high-ERP, short P1 low-ERP)
  Table 3  ERP orthogonalized vs VRP, then VRP+SRP+KRP -> sort the residual
  Table 4  Fama-MacBeth of returns on ERP, adding VRP/SRP/KRP as controls

Uses RAW ERP (not my L_Q normalization) — the paper's method. VRP/SRP/KRP =
physical (realized) minus risk-neutral (BKM) moments, both now in the panel.

    python -m erp.run_paper_tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats.fama_macbeth import fama_macbeth
from .stats.newey_west import nw_mean_tstat

PANEL = "erp/backtest_panel.csv"


def _quartile_ls(panel, sigcol, q=4):
    """Per-month quartile long-short (P_top - P_bottom) + quartile means + NW t."""
    ls, qm = [], np.zeros(q)
    cnt = 0
    for _, d in panel.groupby("date"):
        d = d.dropna(subset=[sigcol, "fwd"])
        if len(d) < 2 * q or d[sigcol].nunique() < q:
            continue
        lab = pd.qcut(d[sigcol], q, labels=range(1, q + 1), duplicates="drop")
        if lab.nunique() < q:
            continue
        m = d.groupby(lab, observed=True)["fwd"].mean()
        ls.append(m.get(q, np.nan) - m.get(1, np.nan))
        qm += np.array([m.get(k, np.nan) for k in range(1, q + 1)])
        cnt += 1
    st = nw_mean_tstat(np.array(ls), lags=5)
    qm = qm / cnt
    mono = all(qm[i] < qm[i + 1] for i in range(q - 1))
    return {"mean_ls": st["mean"], "t": st["t"], "n": cnt, "q_means": qm, "monotonic": mono}


def _orthogonalize(panel, target, controls):
    """Cross-sectional residual of `target` on `controls`, each month."""
    out = pd.Series(np.nan, index=panel.index)
    for _, idx in panel.groupby("date").groups.items():
        d = panel.loc[idx].dropna(subset=[target] + controls)
        if len(d) < len(controls) + 3:
            continue
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in controls])
        b, *_ = np.linalg.lstsq(X, d[target].to_numpy(float), rcond=None)
        out.loc[d.index] = d[target].to_numpy(float) - X @ b
    return out


def main():
    p = pd.read_csv(PANEL)
    p["date"] = pd.to_datetime(p["date"]).dt.date
    print(f"panel: {len(p)} rows, {p['date'].nunique()} months, {p['symbol'].nunique()} names")
    print(f"coverage: erp {p.erp.notna().mean():.0%}  vrp {p.vrp.notna().mean():.0%}  "
          f"srp {p.srp.notna().mean():.0%}  krp {p.krp.notna().mean():.0%}\n")

    print("== Table 2: raw ERP quartile long-short ==")
    for lab, col in [("ERP (raw)", "erp")]:
        r = _quartile_ls(p, col)
        print(f"  {lab:16s} L-S={r['mean_ls']:+.4f} t={r['t']:+.2f} n={r['n']} "
              f"mono={r['monotonic']}  Q1..Q4={['%+.4f'%x for x in r['q_means']]}")

    print("\n== Table 3: ERP orthogonalized, then sorted ==")
    for lab, ctrls in [("ERP perp VRP", ["vrp"]),
                       ("ERP perp VRP,SRP,KRP", ["vrp", "srp", "krp"])]:
        p["resid"] = _orthogonalize(p, "erp", ctrls)
        r = _quartile_ls(p, "resid")
        print(f"  {lab:22s} L-S={r['mean_ls']:+.4f} t={r['t']:+.2f} n={r['n']} "
              f"mono={r['monotonic']}  Q1..Q4={['%+.4f'%x for x in r['q_means']]}")

    print("\n== Table 4: Fama-MacBeth of returns on ERP + controls (ERP slope) ==")
    specs = [("erp",), ("erp", "vrp"), ("erp", "vrp", "srp", "krp")]
    for cols in specs:
        r = fama_macbeth(p, list(cols))
        e = r.get("erp", {})
        extra = ""
        if "vrp" in cols:
            v = r.get("vrp", {})
            extra = f" | vrp t={v.get('t', float('nan')):+.2f}"
        print(f"  {'+'.join(cols):22s} ERP gamma={e.get('gamma', float('nan')):+.5f} "
              f"t={e.get('t', float('nan')):+.2f} p={e.get('p', float('nan')):.3f}{extra}  "
              f"(n={r['n_months']})")


if __name__ == "__main__":
    main()
