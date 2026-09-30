"""Single-name reference data, shared by every security-first page.

The snapshot, the fundamentals page and the MKR framework all want the same
facts about the same ticker. Fetching them page by page would mean three round
trips for one company and — worse — three views that can disagree, because
Yahoo's numbers move between calls. So the fetching and the caching live here,
once, and every caller goes through this service.

Nothing in here raises for a missing optional feed. Fundamentals and option
chains are absent for indices, futures and many non-US listings; a page must
be able to say "not available" and carry on rendering everything else.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.exceptions import DataNotFoundError, MarketIntelError
from market_intel.providers.fundamentals import (
    Fundamentals,
    OptionsSnapshot,
    YFinanceFundamentalsProvider,
)
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

#: Daily history the header quote is computed from — enough for a 52-week band.
QUOTE_HISTORY_DAYS = 400


def optional_feed(
    call: Callable[[], Any], label: str, unavailable: list[str] | None
) -> Any | None:
    """Run an optional enrichment, recording failure instead of raising.

    Args:
        call: The fetch to attempt.
        label: Human-readable feed name, used in the unavailable list.
        unavailable: Collector for failures; ``None`` discards them.

    Returns:
        Whatever ``call`` returned, or ``None`` if it failed.
    """
    try:
        return call()
    except MarketIntelError as exc:
        logger.info("%s unavailable: %s", label, exc)
        if unavailable is not None:
            unavailable.append(f"{label} ({exc})")
    except Exception as exc:  # noqa: BLE001 - an optional feed must never break a page
        logger.warning("%s failed unexpectedly: %s", label, exc)
        if unavailable is not None:
            unavailable.append(f"{label} (unexpected error)")
    return None


@dataclass(frozen=True, slots=True)
class SecurityQuote:
    """The header line for a loaded security: last price and where it sits.

    Deliberately price-only. The company's name and sector come from the
    fundamentals feed, which is optional and much slower, so a quote must not
    depend on it — an index with no fundamentals still has a price.
    """

    symbol: str
    price: float
    as_of: dt.date
    previous_close: float | None = None
    week52_low: float | None = None
    week52_high: float | None = None

    @property
    def change_pct(self) -> float | None:
        """Session move against the prior close, in percent."""
        if self.previous_close in (None, 0):
            return None
        return (self.price / self.previous_close - 1.0) * 100.0

    @property
    def range_position(self) -> float | None:
        """Where the price sits in the 52-week band, 0 at the low, 100 at the high."""
        low, high = self.week52_low, self.week52_high
        if low is None or high is None or high <= low:
            return None
        return (self.price - low) / (high - low) * 100.0


class SecurityService:
    """Quotes, fundamentals and option chains for one ticker at a time."""

    def __init__(
        self,
        market_data: MarketDataService,
        settings: Settings,
        cache: ApiCache | None = None,
        provider: Any | None = None,
    ) -> None:
        self._market_data = market_data
        self._settings = settings
        self._cache = cache
        self._provider = provider or YFinanceFundamentalsProvider()

    # --- Price ----------------------------------------------------------------

    def get_history(
        self, symbol: str, days: int = QUOTE_HISTORY_DAYS, end: dt.date | None = None
    ) -> pd.DataFrame:
        """Daily OHLCV for the trailing ``days``."""
        end = end or dt.date.today()
        return self._market_data.get_price_history(
            symbol.strip().upper(), start=end - dt.timedelta(days=days), end=end
        )

    def get_quote(self, symbol: str, frame: pd.DataFrame | None = None) -> SecurityQuote:
        """Last price plus the 52-week band.

        Args:
            symbol: Ticker to quote.
            frame: Reuse an already-loaded history instead of fetching again.

        Raises:
            DataNotFoundError: No price history exists for the symbol.
        """
        symbol = symbol.strip().upper()
        frame = self.get_history(symbol) if frame is None else frame
        if frame.empty:
            raise DataNotFoundError(f"No price history for '{symbol}'")

        close = frame["close"].astype(float)
        window = close.iloc[-252:]
        return SecurityQuote(
            symbol=symbol,
            price=float(close.iloc[-1]),
            as_of=pd.Timestamp(frame.index[-1]).date(),
            previous_close=float(close.iloc[-2]) if len(close) >= 2 else None,
            week52_low=float(window.min()) if len(window) else None,
            week52_high=float(window.max()) if len(window) else None,
        )

    # --- Optional feeds -------------------------------------------------------

    def get_fundamentals(
        self, symbol: str, unavailable: list[str] | None = None
    ) -> Fundamentals | None:
        """Company facts, cached; ``None`` when the feed has nothing."""
        symbol = symbol.strip().upper()
        key = f"fundamentals:{symbol}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return Fundamentals.from_payload(cached)
        result = optional_feed(
            lambda: self._provider.get_fundamentals(symbol), "fundamentals", unavailable
        )
        if result is not None and self._cache is not None:
            self._cache.set(
                key, result.to_payload(), self._settings.fundamentals_cache_ttl_minutes
            )
        return result

    def get_options(
        self, symbol: str, unavailable: list[str] | None = None
    ) -> OptionsSnapshot | None:
        """Option chain statistics, cached; ``None`` when nothing is listed."""
        symbol = symbol.strip().upper()
        key = f"options:{symbol}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return OptionsSnapshot.from_payload(cached)
        result = optional_feed(
            lambda: self._provider.get_options(symbol), "option chain", unavailable
        )
        if result is not None and self._cache is not None:
            self._cache.set(
                key, result.to_payload(), self._settings.options_cache_ttl_minutes
            )
        return result

    def get_intraday(
        self, symbol: str, unavailable: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Intraday bars for the 4-hour gap scan. Never cached — they age in minutes."""
        return optional_feed(
            lambda: self._provider.get_intraday(symbol.strip().upper()),
            "4-hour fair value gaps",
            unavailable,
        )
