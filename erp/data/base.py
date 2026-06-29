"""Swappable option-data interface (spec Section 4).

Keep loaders interchangeable so the IBKR live loader and any historical vendor
(OptionMetrics/ThetaData) satisfy the same Protocol. The pipeline depends only on
this interface, never on a concrete source — which is also what makes the
pipeline testable against a mock source with no live connection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class OptionChain:
    """One symbol, one expiry, raw (American, for single names) quotes.

    Prices are present-value bids/asks aligned to `strikes`; NaN where missing.
    `tau` is year-fraction to expiry (ACT/365). `spot` is the underlying last.
    """

    symbol: str
    asof: date
    expiry: date
    tau: float
    spot: float
    strikes: np.ndarray
    call_bid: np.ndarray
    call_ask: np.ndarray
    put_bid: np.ndarray
    put_ask: np.ndarray

    def call_mid(self) -> np.ndarray:
        return 0.5 * (self.call_bid + self.call_ask)

    def put_mid(self) -> np.ndarray:
        return 0.5 * (self.put_bid + self.put_ask)


@runtime_checkable
class OptionDataSource(Protocol):
    def get_underlying(self, symbol: str, asof: date) -> float:
        """Spot/last price of the underlying."""
        ...

    def get_expiries(self, symbol: str, asof: date, around_days: int) -> list[date]:
        """Listed expiries, used to bracket the target horizon."""
        ...

    def get_chain(self, symbol: str, asof: date, expiry: date) -> OptionChain:
        ...

    def get_risk_free(self, asof: date, tenor_days: int) -> float:
        """Continuously-compounded risk-free rate r for the tenor."""
        ...

    def get_total_return_history(
        self, symbol: str, asof: date, lookback_days: int
    ) -> np.ndarray:
        """Daily dividend+split-adjusted close series (oldest -> newest), for the
        physical-side realized-return distribution (spec 3e)."""
        ...
