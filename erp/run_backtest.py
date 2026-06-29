"""ERP backtest runner (ThetaData).

    python -m erp.run_backtest --start 2025-01-01 --end 2026-06-26
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

from dotenv import load_dotenv

from .config import Config
from .pipeline.erp_backtest import plot_backtest, run_backtest


def _d(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=_d, default=date(2025, 1, 1))
    ap.add_argument("--end", type=_d, default=date.today())
    ap.add_argument("--symbols", default=None, help="comma list to override universe")
    ap.add_argument("--out", default="erp/backtest.png")
    ap.add_argument("--csv", default="erp/backtest.csv")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.symbols:
        cfg.universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    from .data.thetadata import ThetaDataSource

    print(f"ERP backtest {args.start} .. {args.end}  ({len(cfg.universe)} names)", flush=True)
    src = ThetaDataSource().connect()
    out = run_backtest(src, args.start, args.end, cfg, log=lambda *a: print(*a, flush=True))

    res, summ = out["results"], out.get("summary", {})
    if res.empty:
        print("no valid rebalances"); return
    res.to_csv(args.csv, index=False)
    if out.get("panel") is not None:
        out["panel"].to_csv("erp/backtest_panel.csv", index=False)
        print("saved erp/backtest_panel.csv (per-name L_Q/L_P/erp/fwd)")
    plot_backtest(out, args.out)

    print("\n=== SUMMARY ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")
    print(f"\nsaved {args.csv} and {args.out}")


if __name__ == "__main__":
    main()
