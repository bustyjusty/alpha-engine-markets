"""Tests for the shared single-name service.

The point of this service is that every security-first page reads the same
cached facts, so these tests care most about two things: the quote arithmetic
is right, and an optional feed that fails degrades to ``None`` instead of
taking a page down with it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.providers.base import MarketDataProvider, PriceBar, SecurityInfo
from market_intel.providers.fundamentals import Fundamentals, OptionsSnapshot
from market_intel.services.market_data import MarketDataService
from market_intel.services.security import (
    SecurityQuote,
    SecurityService,
    optional_feed,
)

TODAY = dt.date(2026, 9, 29)


class FakeMarket(MarketDataProvider):
    """A gently rising series with a known start, end and 52-week band."""

    name = "fake"

    def get_security_info(self, symbol: str) -> SecurityInfo:
        return SecurityInfo(symbol=symbol, name=f"{symbol} Corp")

    def get_daily_bars(self, symbol, start, end):
        bars, price = [], 100.0
        day = start
        while day <= end:
            if day.weekday() < 5:
                price *= 1.001
                bars.append(
                    PriceBar(
                        date=day,
                        open=price,
                        high=price * 1.01,
                        low=price * 0.99,
                        close=price,
                        adj_close=price,
                        volume=1_000_000,
                    )
                )
            day += dt.timedelta(days=1)
        return bars


class FakeFeed:
    """Optional feeds that can be told to fail individually."""

    name = "fake"

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.fail = fail
        self.fundamentals_calls = 0
        self.options_calls = 0

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        self.fundamentals_calls += 1
        if "fundamentals" in self.fail:
            raise DataNotFoundError("no coverage", provider=self.name)
        return Fundamentals(symbol=symbol, name=f"{symbol} Inc", currency="USD")

    def get_options(self, symbol: str) -> OptionsSnapshot:
        self.options_calls += 1
        if "options" in self.fail:
            raise DataNotFoundError("no listed options", provider=self.name)
        return OptionsSnapshot(symbol=symbol, as_of=TODAY, max_pain=105.0)

    def get_intraday(self, symbol: str, days: int = 60) -> pd.DataFrame:
        if "intraday" in self.fail:
            raise ProviderError("intraday down", provider=self.name)
        return pd.DataFrame({"close": [1.0, 2.0]})


@pytest.fixture()
def service() -> SecurityService:
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(_env_file=None)
    cache = ApiCache(db)
    market_data = MarketDataService(db, FakeMarket(), cache, settings)
    return SecurityService(market_data, settings, cache, FakeFeed())


# --- Quote arithmetic ---------------------------------------------------------


def test_change_pct_is_measured_against_the_prior_close() -> None:
    quote = SecurityQuote(
        symbol="X", price=110.0, as_of=TODAY, previous_close=100.0
    )
    assert quote.change_pct == pytest.approx(10.0)


def test_change_pct_is_none_without_a_prior_close() -> None:
    quote = SecurityQuote(symbol="X", price=110.0, as_of=TODAY)
    assert quote.change_pct is None


def test_a_zero_prior_close_does_not_divide_by_zero() -> None:
    """A delisted or malformed bar must not raise on the header of every page."""
    quote = SecurityQuote(
        symbol="X", price=110.0, as_of=TODAY, previous_close=0.0
    )
    assert quote.change_pct is None


def test_range_position_places_the_price_in_the_band() -> None:
    quote = SecurityQuote(
        symbol="X", price=75.0, as_of=TODAY, week52_low=50.0, week52_high=150.0
    )
    assert quote.range_position == pytest.approx(25.0)


def test_range_position_is_none_when_the_band_is_degenerate() -> None:
    quote = SecurityQuote(
        symbol="X", price=50.0, as_of=TODAY, week52_low=50.0, week52_high=50.0
    )
    assert quote.range_position is None


def test_get_quote_reads_the_last_bar_and_the_52_week_band(service) -> None:
    quote = service.get_quote("NVDA")
    frame = service.get_history("NVDA")
    closes = frame["close"].astype(float)

    assert quote.symbol == "NVDA"
    assert quote.price == pytest.approx(float(closes.iloc[-1]))
    assert quote.previous_close == pytest.approx(float(closes.iloc[-2]))
    # The series only rises, so the window's last bar is also its high.
    assert quote.week52_high == pytest.approx(float(closes.iloc[-252:].max()))
    assert quote.change_pct > 0


def test_get_quote_reuses_a_supplied_frame(service) -> None:
    """Pages that already loaded history must not trigger a second read."""
    frame = service.get_history("NVDA")
    assert service.get_quote("NVDA", frame).price == pytest.approx(
        float(frame["close"].astype(float).iloc[-1])
    )


def test_get_quote_raises_when_there_is_no_history(service, monkeypatch) -> None:
    monkeypatch.setattr(
        service, "get_history", lambda *args, **kwargs: pd.DataFrame()
    )
    with pytest.raises(DataNotFoundError):
        service.get_quote("NOPE")


# --- Optional feeds -----------------------------------------------------------


def test_optional_feed_records_failure_instead_of_raising() -> None:
    unavailable: list[str] = []
    result = optional_feed(
        lambda: (_ for _ in ()).throw(DataNotFoundError("gone")),
        "fundamentals",
        unavailable,
    )
    assert result is None
    assert unavailable and unavailable[0].startswith("fundamentals")


def test_optional_feed_survives_an_unexpected_error() -> None:
    """A provider raising something outside the hierarchy must still not escape."""
    unavailable: list[str] = []
    assert optional_feed(lambda: 1 / 0, "option chain", unavailable) is None
    assert unavailable == ["option chain (unexpected error)"]


def test_optional_feed_accepts_no_collector() -> None:
    assert optional_feed(lambda: 1 / 0, "whatever", None) is None


def test_missing_fundamentals_degrade_to_none(service) -> None:
    service._provider = FakeFeed(fail=("fundamentals",))
    unavailable: list[str] = []
    assert service.get_fundamentals("SPX", unavailable) is None
    assert unavailable == ["fundamentals (no coverage)"]


def test_missing_options_degrade_to_none(service) -> None:
    service._provider = FakeFeed(fail=("options",))
    unavailable: list[str] = []
    assert service.get_options("ASML.AS", unavailable) is None
    assert unavailable == ["option chain (no listed options)"]


def test_missing_intraday_degrades_to_none(service) -> None:
    service._provider = FakeFeed(fail=("intraday",))
    assert service.get_intraday("NVDA") is None


# --- Caching ------------------------------------------------------------------


def test_fundamentals_are_fetched_once_and_then_cached(service) -> None:
    feed = service._provider
    first = service.get_fundamentals("NVDA")
    second = service.get_fundamentals("NVDA")

    assert feed.fundamentals_calls == 1
    assert first == second


def test_options_are_fetched_once_and_then_cached(service) -> None:
    feed = service._provider
    assert service.get_options("NVDA").max_pain == 105.0
    assert service.get_options("NVDA").max_pain == 105.0
    assert feed.options_calls == 1


def test_the_cache_is_keyed_per_symbol(service) -> None:
    service.get_fundamentals("NVDA")
    service.get_fundamentals("AMD")
    assert service._provider.fundamentals_calls == 2


def test_symbols_are_normalised_before_lookup(service) -> None:
    """`nvda `, `NVDA` and ` Nvda` are one security, and so one cache entry."""
    service.get_fundamentals(" nvda ")
    service.get_fundamentals("NVDA")
    assert service._provider.fundamentals_calls == 1
