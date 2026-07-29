"""Market data adapter for Yahoo Finance (via the ``yfinance`` package).

Free and keyless, which makes it the default provider. Yahoo data is
unofficial; the ``source`` column stored with every bar records that the
numbers came from here.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from typing import Any

import pandas as pd
import yfinance as yf

from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.models import (
    CalendarEvent,
    EtfHoldingItem,
    NewsItem,
    PriceBar,
    SecurityInfo,
)
from market_intel.providers.base import (
    EtfHoldingsProvider,
    EventProvider,
    MarketDataProvider,
    NewsProvider,
)

logger = logging.getLogger(__name__)

_ASSET_TYPES = {
    "EQUITY": "equity",
    "ETF": "etf",
    "INDEX": "index",
    "MUTUALFUND": "fund",
    "CURRENCY": "fx",
    "CRYPTOCURRENCY": "crypto",
    "FUTURE": "commodity",
}


class YFinanceProvider(MarketDataProvider):
    """Reference data and daily OHLCV bars from Yahoo Finance."""

    name = "yfinance"

    def get_security_info(self, symbol: str) -> SecurityInfo:
        try:
            info: dict[str, Any] = yf.Ticker(symbol).get_info()
        except Exception as exc:  # yfinance raises assorted internal errors
            raise ProviderError(
                f"yfinance info request failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc

        if not info or info.get("quoteType") in (None, "NONE"):
            raise DataNotFoundError(
                f"No security info found for '{symbol}'", provider=self.name
            )

        return SecurityInfo(
            symbol=symbol.strip().upper(),
            name=info.get("longName") or info.get("shortName"),
            asset_type=_ASSET_TYPES.get(info.get("quoteType", ""), "equity"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency=info.get("currency"),
            exchange=info.get("exchange"),
        )

    def get_daily_bars(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[PriceBar]:
        try:
            frame = yf.Ticker(symbol).history(
                start=start.isoformat(),
                # yfinance treats `end` as exclusive.
                end=(end + dt.timedelta(days=1)).isoformat(),
                interval="1d",
                auto_adjust=False,
            )
        except Exception as exc:
            raise ProviderError(
                f"yfinance history request failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc

        if frame.empty:
            raise DataNotFoundError(
                f"No bars for '{symbol}' between {start} and {end}",
                provider=self.name,
            )
        return frame_to_bars(frame, source=self.name)


def frame_to_bars(frame: pd.DataFrame, source: str) -> list[PriceBar]:
    """Convert a yfinance history frame into domain bars.

    Rows without a close price (halts, bad rows) are skipped.
    """
    bars: list[PriceBar] = []
    for timestamp, row in frame.iterrows():
        close = _as_float(row.get("Close"))
        if close is None:
            continue
        bars.append(
            PriceBar(
                date=timestamp.date(),
                close=close,
                open=_as_float(row.get("Open")),
                high=_as_float(row.get("High")),
                low=_as_float(row.get("Low")),
                adj_close=_as_float(row.get("Adj Close")),
                volume=_as_int(row.get("Volume")),
                source=source,
            )
        )
    return bars


class YFinanceNewsProvider(NewsProvider):
    """Recent headlines from Yahoo Finance, per symbol."""

    name = "yfinance"

    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        try:
            raw_items = yf.Ticker(symbol).get_news(count=limit)
        except Exception as exc:
            raise ProviderError(
                f"yfinance news request failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc

        items = []
        for raw in raw_items or []:
            item = parse_news_item(raw, symbol=symbol.strip().upper())
            if item is not None:
                items.append(item)
        return items


def parse_news_item(raw: dict[str, Any], symbol: str) -> NewsItem | None:
    """Parse one yfinance news dict into a domain item.

    Handles both the current nested format (``{"content": {...}}``) and the
    legacy flat format (``{"title", "link", "providerPublishTime"}``).
    Returns None for items missing a headline or URL.
    """
    content = raw.get("content") or raw

    headline = content.get("title")
    url = (
        (content.get("canonicalUrl") or {}).get("url")
        if isinstance(content.get("canonicalUrl"), dict)
        else None
    ) or content.get("link")
    if not headline or not url:
        return None

    source = None
    provider_field = content.get("provider") or content.get("publisher")
    if isinstance(provider_field, dict):
        source = provider_field.get("displayName")
    elif isinstance(provider_field, str):
        source = provider_field

    published_at = None
    pub_date = content.get("pubDate")
    if isinstance(pub_date, str):
        try:
            published_at = dt.datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        except ValueError:
            pass
    elif isinstance(raw.get("providerPublishTime"), (int, float)):
        published_at = dt.datetime.fromtimestamp(
            raw["providerPublishTime"], tz=dt.timezone.utc
        )

    return NewsItem(
        headline=headline,
        url=url,
        source=source,
        summary=content.get("summary"),
        published_at=published_at,
        symbols=(symbol,),
    )


class YFinanceEventProvider(EventProvider):
    """Earnings dates from Yahoo Finance."""

    name = "yfinance"

    def get_earnings_events(self, symbol: str) -> list[CalendarEvent]:
        try:
            frame = yf.Ticker(symbol).get_earnings_dates(limit=8)
        except Exception as exc:
            raise ProviderError(
                f"yfinance earnings dates failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc
        if frame is None or frame.empty:
            return []
        return earnings_frame_to_events(frame, symbol=symbol.strip().upper())


def earnings_frame_to_events(frame: pd.DataFrame, symbol: str) -> list[CalendarEvent]:
    """Convert a yfinance earnings-dates frame into calendar events.

    The frame is indexed by (tz-aware) earnings timestamp with columns
    like ``EPS Estimate`` and ``Reported EPS``.
    """
    events: list[CalendarEvent] = []
    for timestamp, row in frame.iterrows():
        scheduled_at = timestamp.to_pydatetime()
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=dt.timezone.utc)
        events.append(
            CalendarEvent(
                event_type="earnings",
                title=f"{symbol} earnings",
                scheduled_at=scheduled_at,
                symbol=symbol,
                consensus=_as_float(row.get("EPS Estimate")),
                actual=_as_float(row.get("Reported EPS")),
                source="yfinance",
            )
        )
    return events


class YFinanceEtfProvider(EtfHoldingsProvider):
    """Top ETF constituents from Yahoo Finance.

    Yahoo exposes only the top ~10 holdings per fund; that covers the
    concentration that matters for exposure mapping, but weights will not
    sum to 1. A full-holdings provider (e.g. issuer files) can be added
    behind the same interface later.
    """

    name = "yfinance"

    def get_etf_holdings(self, symbol: str) -> list[EtfHoldingItem]:
        try:
            frame = yf.Ticker(symbol).funds_data.top_holdings
        except Exception as exc:
            raise ProviderError(
                f"yfinance holdings failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc
        if frame is None or frame.empty:
            raise DataNotFoundError(
                f"No holdings data for '{symbol}' (not an ETF?)", provider=self.name
            )
        return holdings_frame_to_items(frame)


def holdings_frame_to_items(frame: pd.DataFrame) -> list[EtfHoldingItem]:
    """Convert a yfinance top-holdings frame (indexed by symbol) to items."""
    items: list[EtfHoldingItem] = []
    for holding_symbol, row in frame.iterrows():
        items.append(
            EtfHoldingItem(
                holding_symbol=str(holding_symbol).strip().upper(),
                weight=_as_float(row.get("Holding Percent")),
                holding_name=row.get("Name"),
            )
        )
    items.sort(key=lambda item: item.weight or 0.0, reverse=True)
    return items


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)
