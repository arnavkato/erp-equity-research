"""Fama-MacBeth test of ERP on the Dow 30, on the validated entropy panel.

Rebuilds L^P with the cumulant estimator (validated; less noisy), forms the raw
and vol-normalized ERP, and runs FMB (the paper's statistic) — using every name
every month instead of 7-name quartiles.

    python -m erp.run_fama_macbeth
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .config import Config
from .core.entropy import physical_entropy_cumulant
from .pipeline.erp_snapshot import TRADING_DAYS_PER_YEAR
from .stats.fama_macbeth import fama_macbeth

PANEL = "erp/backtest_panel.csv"


def _dlr(prices):
    p = np.asarray(prices, float)
    p = p[np.isfinite(p) & (p > 0)]
    return np.diff(np.log(p)) if p.size > 2 else np.array([])


def main():
    load_dotenv()
    panel = pd.read_csv(PANEL)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    cfg = Config()
    hz = round(cfg.target_tau_days * TRADING_DAYS_PER_YEAR / 365)

    from .data.thetadata import ThetaDataSource
    src = ThetaDataSource().connect()

    # rebuild L^P with the validated cumulant estimator (Gaussian order kept —
    # higher realized moments were pure noise on this universe)
    lp = {}
    for (sym, d), _ in panel.groupby(["symbol", "date"]):
        lp[(sym, d)] = physical_entropy_cumulant(
            _dlr(src.get_total_return_history(sym, d, 365)), hz, max_order=2)
    panel["L_P_cum"] = [lp.get((s, d), np.nan) for s, d in zip(panel["symbol"], panel["date"])]
    panel["erp_cum"] = panel["L_P_cum"] - panel["L_Q"]
    panel["erp_norm"] = panel["erp_cum"] / panel["L_Q"]

    print(f"panel: {len(panel)} rows, {panel['date'].nunique()} months, "
          f"{panel['symbol'].nunique()} names\n")

    print("Fama-MacBeth (slope = mean fwd return per 1 cross-sectional SD, NW 5-lag):")
    for label, col in [("raw ERP (cumulant L_P)", "erp_cum"),
                       ("raw ERP (orig overlap L_P)", "erp"),
                       ("vol-normalized ERP/L_Q (cumulant)", "erp_norm")]:
        r = fama_macbeth(panel, col)
        s = r.get(col, {})
        print(f"  {label:38s}  gamma={s.get('gamma', float('nan')):+.5f}  "
              f"t={s.get('t', float('nan')):+.2f}  p={s.get('p', float('nan')):.3f}  "
              f"months={r['n_months']}")

    # multivariate: does normalized ERP survive controlling for the vol level L_Q?
    r = fama_macbeth(panel, ["erp_norm", "L_Q"])
    print("\nMultivariate FMB (normalized ERP + L_Q level as control):")
    for c in ("erp_norm", "L_Q"):
        s = r[c]
        print(f"  {c:12s}  gamma={s['gamma']:+.5f}  t={s['t']:+.2f}  p={s['p']:.3f}")


if __name__ == "__main__":
    main()
