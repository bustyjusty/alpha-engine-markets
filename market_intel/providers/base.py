"""Abstract provider interfaces.

Adapters return domain models and raise the platform's typed exceptions —
no provider payloads or provider-specific errors may leak past this
boundary. That contract is what makes providers swappable.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod

from market_intel.models import (
    CalendarEvent,
    EtfHoldingItem,
    NewsItem,
    PriceBar,
    SecurityInfo,
)


class MarketDataProvider(ABC):
    """Source of security reference data and daily price history."""

    #: Registry name; also recorded as the ``source`` of fetched data.
    name: str = "abstract"

    @abstractmethod
    def get_security_info(self, symbol: str) -> SecurityInfo:
        """Return reference data for a symbol.

        Raises:
            DataNotFoundError: The symbol is unknown to this provider.
            ProviderError: The request failed for any other reason.
        """

    @abstractmethod
    def get_daily_bars(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[PriceBar]:
        """Return daily bars for ``start``..``end`` inclusive, oldest first.

        Raises:
            DataNotFoundError: No bars exist in the range (e.g. weekends only).
            ProviderError: The request failed for any other reason.
        """


class NewsProvider(ABC):
    """Source of financial news headlines."""

    name: str = "abstract"

    @abstractmethod
    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        """Return recent news for a symbol, newest first.

        Returns an empty list when the symbol simply has no coverage.

        Raises:
            ProviderError: The request failed.
        """


class EventProvider(ABC):
    """Source of dated market events (earnings dates, macro releases)."""

    name: str = "abstract"

    @abstractmethod
    def get_earnings_events(self, symbol: str) -> list[CalendarEvent]:
        """Return recent and upcoming earnings events for a symbol.

        Returns an empty list when no earnings dates are available.

        Raises:
            ProviderError: The request failed.
        """


class EtfHoldingsProvider(ABC):
    """Source of ETF constituent data."""

    name: str = "abstract"

    @abstractmethod
    def get_etf_holdings(self, symbol: str) -> list[EtfHoldingItem]:
        """Return current constituents of an ETF, largest weight first.

        Raises:
            DataNotFoundError: The symbol is not an ETF or has no holdings data.
            ProviderError: The request failed for any other reason.
        """
