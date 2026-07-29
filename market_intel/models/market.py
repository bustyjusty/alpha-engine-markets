"""Domain models for market data."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SecurityInfo:
    """Static reference data describing a tradable security.

    Only ``symbol`` is required; providers fill in whatever metadata they
    have and repositories merge non-null fields into the stored record.
    """

    symbol: str
    name: str | None = None
    asset_type: str = "equity"  # equity | etf | index | fx | commodity
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    exchange: str | None = None


@dataclass(frozen=True, slots=True)
class PriceBar:
    """A single daily OHLCV bar.

    ``adj_close`` is the split/dividend-adjusted close when the provider
    supplies one; analytics should prefer it and fall back to ``close``.
    """

    date: dt.date
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    adj_close: float | None = None
    volume: int | None = None
    source: str = "unknown"
