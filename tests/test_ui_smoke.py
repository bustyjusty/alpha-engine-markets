"""Smoke tests: every dashboard page renders without an exception.

Uses Streamlit's AppTest with the service graph swapped for fakes, so no
network is touched.
"""

import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

import ui.context
from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import DataNotFoundError
from market_intel.models import CalendarEvent, EtfHoldingItem, NewsItem, PriceBar, SecurityInfo
from market_intel.providers.base import (
    EtfHoldingsProvider,
    EventProvider,
    MarketDataProvider,
    NewsProvider,
)
from market_intel.services import MarketDataService, NewsService
from market_intel.services.calendar import CalendarService
from market_intel.services.etf import EtfService
from market_intel.services.portfolio import JournalService, ThemeService, WatchlistService
from market_intel.services.research import ResearchService
from market_intel.services.scanner import ScannerService

PAGES = [
    "ui/pages/overview.py",
    "ui/pages/charts.py",
    "ui/pages/news.py",
    "ui/pages/calendar_page.py",
    "ui/pages/scanner.py",
    "ui/pages/etf_explorer.py",
    "ui/pages/themes.py",
    "ui/pages/research.py",
    "ui/pages/journal.py",
]


class FakeMarket(MarketDataProvider):
    name = "fake"

    def get_security_info(self, symbol):
        return SecurityInfo(symbol=symbol, name=f"{symbol} Corp")

    def get_daily_bars(self, symbol, start, end):
        bars, price = [], 100.0
        for i in range((end - start).days + 1):
            price *= 1.001 if i % 2 else 0.9995
            bars.append(
                PriceBar(
                    date=start + dt.timedelta(days=i),
                    close=price,
                    open=price * 0.99,
                    high=price * 1.01,
                    low=price * 0.98,
                    adj_close=price,
                    volume=1_000_000,
                    source="fake",
                )
            )
        return bars


class FakeNews(NewsProvider):
    name = "fake"

    def get_news(self, symbol, limit=20):
        return [NewsItem(headline=f"{symbol} in the news", url=f"https://x/{symbol}")]


class FakeEvents(EventProvider):
    name = "fake"

    def get_earnings_events(self, symbol):
        return [
            CalendarEvent(
                event_type="earnings",
                title=f"{symbol} earnings",
                scheduled_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=7),
                symbol=symbol,
            )
        ]


class FakeEtf(EtfHoldingsProvider):
    name = "fake"

    def get_etf_holdings(self, symbol):
        if symbol != "SMH":
            raise DataNotFoundError("not an etf", provider=self.name)
        return [EtfHoldingItem("NVDA", weight=0.2, holding_name="NVIDIA")]


@pytest.fixture()
def fake_services(monkeypatch):
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(_env_file=None)
    cache = ApiCache(db)
    market_data = MarketDataService(db, FakeMarket(), cache, settings)
    services = ui.context.AppServices(
        settings=settings,
        db=db,
        market_data=market_data,
        news=NewsService(db, FakeNews(), cache, settings, market_data),
        calendar=CalendarService(db, FakeEvents(), cache, settings, market_data),
        etf=EtfService(db, FakeEtf(), cache, settings, market_data),
        scanner=ScannerService(market_data),
        research=ResearchService(db, settings),
        watchlists=WatchlistService(db, market_data),
        themes=ThemeService(db, market_data),
        journal=JournalService(db, market_data),
    )
    monkeypatch.setattr(ui.context, "get_services", lambda: services)
    return services


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str, fake_services) -> None:
    at = AppTest.from_file(page, default_timeout=30)
    at.run()
    assert not at.exception, at.exception


def test_overview_with_populated_watchlist(fake_services) -> None:
    fake_services.watchlists.add_symbol("Core", "NVDA")
    at = AppTest.from_file("ui/pages/overview.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert at.dataframe  # returns table rendered
