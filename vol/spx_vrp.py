"""SPX variance risk premium, 30-day rolling. Implied = VIX (it already is the
30d risk-neutral vol of SPX); realized = trailing 30-trading-day realized vol.
VRP = implied variance - realized variance (annualized).

    python vol/spx_vrp.py

ponytail: realized is TRAILING 30d (what "rolling" means + the standard chart).
For the economically-correct seller's VRP, swap to FORWARD realized over the next
30d: rv = r.pow(2).rolling(WIN).sum().shift(-WIN)*(252/WIN).
"""
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from thetadata import ThetaClient

WIN = 30
load_dotenv("/workspaces/app/.env")
tc = ThetaClient(email=os.getenv("THETA_EMAIL"), password=os.getenv("THETA_PASSWORD"),
                 dataframe_type="pandas")


def idx(sym, start, end):
    frames, cur = [], start
    while cur <= end:
        ce = min(end, cur + timedelta(days=360))
        d = tc.index_history_eod(sym, cur, ce)
        if d is not None and len(d):
            frames.append(d)
        cur = ce + timedelta(days=1)
    df = pd.concat(frames, ignore_index=True)
    s = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["created"]).dt.normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


start, end = date(2024, 1, 1), date.today()
vix, spx = idx("VIX", start, end), idx("SPX", start, end)
r = np.log(spx).diff()
rv = r.pow(2).rolling(WIN).sum() * (252 / WIN)          # annualized realized variance, trailing 30d
df = pd.DataFrame({"vix": vix, "rvol": np.sqrt(rv) * 100}).dropna()
df["ivar"] = (df["vix"] / 100) ** 2
df["rvar"] = (df["rvol"] / 100) ** 2
df["vrp"] = df["ivar"] - df["rvar"]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1])
a1.plot(df.index, df.vix, color="#d62728", label="VIX (30d implied vol)")
a1.plot(df.index, df.rvol, color="#1f77b4", label="realized vol (rolling 30d)")
a1.set_ylabel("vol %"); a1.legend(); a1.grid(alpha=0.3)
a1.set_title(f"SPX variance risk premium — 30-day rolling  "
             f"(mean implied−realized {(df.vix-df.rvol).mean():+.1f} vol pts, VRP>0 {(df.vrp>0).mean():.0%} of days)")
a2.fill_between(df.index, df.vrp, 0, where=df.vrp >= 0, color="#2ca02c", alpha=0.5)
a2.fill_between(df.index, df.vrp, 0, where=df.vrp < 0, color="#d62728", alpha=0.5)
a2.axhline(0, color="k", lw=0.7); a2.set_ylabel("VRP (impl var − real var, ann.)"); a2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("vol/spx_vrp.png", dpi=110)
print(f"saved vol/spx_vrp.png | {df.index[0].date()}..{df.index[-1].date()} "
      f"| mean VRP {df.vrp.mean():+.5f} var | mean impl−real {(df.vix-df.rvol).mean():+.1f} vol pts "
      f"| VRP>0 {(df.vrp>0).mean():.0%} of days")
