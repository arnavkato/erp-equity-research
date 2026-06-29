"""Configuration: universe, horizon, filters (spec config.py)."""
from __future__ import annotations

from dataclasses import dataclass, field

# Dow Jones Industrial Average constituents (as of 2026; NVDA/SHW added Nov 2024,
# AMZN added Feb 2024). Verify membership before a live run — the index reconstitutes.
DOW_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]


@dataclass
class Config:
    universe: list = field(default_factory=lambda: list(DOW_30))

    # Constant-maturity target (calendar days). The strip is computed for the two
    # bracketing expiries and interpolated to this horizon (spec 3c).
    target_tau_days: int = 30

    # Physical side (spec 3d/3e). Per the paper, L^P proxies the physical risk
    # measure with realized moments over a ~60-trading-day rolling window
    # (daily-return cumulants, scaled to the option horizon). Keeps L^P on the
    # same timescale as the forward-looking L^Q. physical_fetch_days is the
    # calendar history pulled to yield physical_window_td trading days.
    physical_window_td: int = 60
    physical_cumulant_order: int = 2     # 2=Gaussian; 3/4 add realized skew/kurt
    physical_fetch_days: int = 120

    # Filters (spec Section 5).
    min_tick: float = 0.01
    min_price_ticks: float = 5.0      # discard options priced below 5x min tick
    min_otm_per_side: int = 2          # need >= 2 OTM puts and >= 2 OTM calls
    max_rel_spread: float = 0.75       # drop quotes with absurdly wide bid-ask

    # Integration grid (spec Section 5.4).
    n_std: float = 8.0
    n_grid: int = 2000

    # De-Americanization carry iteration (spec 3b). 1 is plenty — the parity
    # forward barely moves on a second pass for short-dated equity options.
    carry_iters: int = 1

    # IBKR pacing (spec Section 4).
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002               # IB Gateway paper; 7497 TWS paper, 4001/7496 live
    ib_client_id: int = 17
    md_batch_size: int = 50           # <= 100 concurrent market-data lines
    md_dwell_secs: float = 8.0         # let delayed ticks arrive before reading
    # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen. Accounts without a real-time
    # API options entitlement must use delayed (the gateway offers it for free).
    market_data_type: int = 3
    # Without an OPRA bid/ask entitlement, fall back to settlement/last prices so
    # the strip can still be built. Makes the snapshot EOD-settlement, not live-mid.
    settlement_fallback: bool = True
