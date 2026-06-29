"""VIX reconstruction gate (spec Section 7 — the critical correctness check).

The risk-neutral log-contract strip we build for L^Q is the same 1/K^2 OTM-option
strip the VIX/variance-swap is built on. So feed real SPX chains into our strip
code and reconstruct the *published* VIX. If we can't match it to ~a few tenths
of a vol point, the integration/interpolation is wrong and every ERP number is
suspect. SPX options are European, so this tests the strip alone (no de-Am).

    var_tau(annualized) = 2 * strip / tau          (strip = R_f * int O/K^2 dK)
    VIX = 100 * sqrt( var interpolated to 30 days )

    python -m erp.vix_check --dates 2026-06-25,2026-03-20,2025-11-21
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .core.entropy import log_contract_strip
from .core.forward import implied_forward
from .core.smile import OTMQuotes


def _chain_var(tc, asof, exp, r):
    """Annualized risk-neutral variance from one SPXW expiry via our strip."""
    df = tc.option_history_eod(asof, asof, "SPXW", exp, strike="*", right="both")
    if df is None or df.empty:
        return None
    df = df.copy()
    df["right"] = df["right"].str.upper()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & np.isfinite(df["mid"])]
    piv = df.pivot_table(index="strike", columns="right", values="mid")
    if "CALL" not in piv or "PUT" not in piv:
        return None
    K = piv.index.to_numpy(float)
    cmid, pmid = piv["CALL"].to_numpy(float), piv["PUT"].to_numpy(float)
    tau = (exp - asof).days / 365.0

    both = np.isfinite(cmid) & np.isfinite(pmid)
    F, _ = implied_forward(K[both], cmid[both], pmid[both], r, tau)
    if not np.isfinite(F) or F <= 0:
        return None

    is_call = K > F
    otm = np.where(is_call, cmid, pmid)
    keep = np.isfinite(otm) & (otm > 0) & (K > 0)
    if keep.sum() < 6:
        return None
    q = OTMQuotes(F=float(F), r=r, tau=tau, strikes=K[keep], prices=otm[keep], is_call=is_call[keep])
    strip = log_contract_strip(q, n_std=10.0, n_grid=4000)
    return 2.0 * strip / tau, tau  # annualized variance, year-fraction


def reconstruct_vix(tc, asof) -> dict:
    exps = pd.to_datetime(tc.option_list_expirations("SPXW")["expiration"].astype(str)).dt.date
    lo = max((e for e in exps if 23 <= (e - asof).days <= 30), default=None)
    hi = min((e for e in exps if 30 <= (e - asof).days <= 37), default=None)
    if lo is None or hi is None:
        return {"asof": asof, "vix_recon": np.nan, "note": "no 23-37d bracket"}
    rdf = tc.interest_rate_history_eod("SOFR", asof, asof)
    r = float(rdf.iloc[-1]["rate"]) / 100.0 if rdf is not None and not rdf.empty else 0.045

    out = {}
    for tag, e in (("lo", lo), ("hi", hi)):
        res = _chain_var(tc, asof, e, r)
        if res is None:
            return {"asof": asof, "vix_recon": np.nan, "note": f"{tag} chain failed"}
        out[tag] = res
    (v1, t1), (v2, t2) = out["lo"], out["hi"]
    # interpolate annualized variance to 30 calendar days
    target = 30 / 365.0
    w = (t2 - target) / (t2 - t1) if t2 != t1 else 0.5
    var30 = w * v1 + (1 - w) * v2
    vix_recon = 100.0 * np.sqrt(max(var30, 0.0))

    vdf = tc.index_history_eod("VIX", asof, asof)
    vix_pub = float(vdf.iloc[-1]["close"]) if vdf is not None and not vdf.empty else np.nan
    return {"asof": asof, "dte_lo": (lo - asof).days, "dte_hi": (hi - asof).days,
            "vix_recon": round(vix_recon, 2), "vix_published": round(vix_pub, 2),
            "diff": round(vix_recon - vix_pub, 2)}


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="2026-06-25,2026-03-20,2025-11-21,2025-08-15,2025-04-17")
    args = ap.parse_args(argv)
    dates = [datetime.strptime(s.strip(), "%Y-%m-%d").date() for s in args.dates.split(",")]

    from thetadata import ThetaClient
    tc = ThetaClient(email=os.getenv("THETA_EMAIL"), password=os.getenv("THETA_PASSWORD"),
                     dataframe_type="pandas")
    rows = [reconstruct_vix(tc, d) for d in dates]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
