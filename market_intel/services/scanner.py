"""Relative-value scanner service.

Pulls price history through :class:`MarketDataService` (so data is
fetched/persisted as needed) and delegates the math to the pure
:func:`~market_intel.analysis.relative_value.relative_value_scan`.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

import pandas as pd

from market_intel.analysis import indicator_snapshot, relative_value_scan
from market_intel.exceptions import ProviderError
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class ScannerService:
    """Ranks a universe of symbols by unusualness of their recent move."""

    def __init__(self, market_data: MarketDataService) -> None:
        self._market_data = market_data

    def scan(
        self,
        symbols: Sequence[str],
        benchmark_symbol: str = "SPY",
        window_days: int = 5,
        lookback_days: int = 180,
        zscore_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """Scan symbols against a benchmark over a reaction window.

        Symbols whose data cannot be fetched are skipped with a warning.

        Returns:
            DataFrame indexed by symbol with the relative-value columns
            (see ``relative_value_scan``) plus per-symbol technical
            context: ``last_close``, ``rsi``, ``vwap_dev`` (distance from
            the 20-session rolling VWAP), ``percent_b`` (Bollinger) and a
            comma-joined ``signals`` column (oversold/overbought, VWAP
            stretch, band breaches, MACD crosses). Empty if the benchmark
            itself is unavailable.
        """
        start = dt.date.today() - dt.timedelta(days=lookback_days)

        benchmark_symbol = benchmark_symbol.strip().upper()
        try:
            benchmark = _closes(self._fetch(benchmark_symbol, start))
        except ProviderError as exc:
            logger.error("Benchmark %s unavailable: %s", benchmark_symbol, exc)
            return pd.DataFrame()

        frames: dict[str, pd.DataFrame] = {}
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper()
            if symbol == benchmark_symbol:
                continue
            try:
                frames[symbol] = self._fetch(symbol, start)
            except ProviderError as exc:
                logger.warning("Skipping %s in scan: %s", symbol, exc)

        if not frames:
            return pd.DataFrame()

        result = relative_value_scan(
            {symbol: _closes(frame) for symbol, frame in frames.items()},
            benchmark,
            window_days=window_days,
            zscore_threshold=zscore_threshold,
        )

        # Enrich each row with a technical-indicator snapshot.
        snapshots = {symbol: indicator_snapshot(frame) for symbol, frame in frames.items()}
        for column in ("close", "rsi", "vwap_dev", "percent_b"):
            result[column if column != "close" else "last_close"] = [
                snapshots[symbol][column] for symbol in result.index
            ]
        result["signals"] = [
            ", ".join(snapshots[symbol]["signals"]) for symbol in result.index
        ]
        return result

    def _fetch(self, symbol: str, start: dt.date) -> pd.DataFrame:
        return self._market_data.get_price_history(symbol, start=start)


def _closes(frame: pd.DataFrame) -> pd.Series:
    """Adjusted closes (fallback: raw closes)."""
    if frame.empty:
        return pd.Series(dtype=float)
    return frame["adj_close"].fillna(frame["close"]).astype(float)
