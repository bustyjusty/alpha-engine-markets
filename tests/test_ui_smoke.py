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
from market_intel.providers.fundamentals import Fundamentals, OptionQuote, OptionsSnapshot
from market_intel.providers.base import (
    EtfHoldingsProvider,
    EventProvider,
    MarketDataProvider,
    NewsProvider,
)
from market_intel.services import MarketDataService, NewsService
from market_intel.services.calendar import CalendarService
from market_intel.services.etf import EtfService
from market_intel.services.mkr import MkrService
from market_intel.services.portfolio import JournalService, ThemeService, WatchlistService
from market_intel.services.recap import RecapService
from market_intel.services.report import ReportPipeline
from market_intel.services.research import ResearchService
from market_intel.services.scanner import ScannerService
from market_intel.universe import AssetClass, Instrument, QuoteKind, Region

PAGES = [
    "ui/pages/overview.py",
    "ui/pages/recap.py",
    "ui/pages/mkr.py",
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


class FakeFundamentalsFeed:
    """Stands in for the optional single-name feeds the MKR page consumes."""

    name = "fake"

    def get_fundamentals(self, symbol):
        return Fundamentals(
            symbol=symbol,
            name=f"{symbol} Corp",
            market_cap=1.2e10,
            peg_ratio=0.9,
            profit_margin=0.18,
            revenue_growth=0.25,
            target_low=90.0,
            target_mean=140.0,
            target_high=180.0,
            analyst_count=12,
            next_earnings=dt.date.today() + dt.timedelta(days=30),
        )

    def get_options(self, symbol, max_expiries=6):
        near = dt.date.today() + dt.timedelta(days=35)
        far = dt.date.today() + dt.timedelta(days=280)
        quotes = [
            OptionQuote(expiry, strike, kind, strike * 0.05, strike * 0.055,
                        strike * 0.052, 0.4, 500, 100)
            for expiry in (near, far)
            for strike in (80.0, 100.0, 120.0, 140.0)
            for kind in ("call", "put")
        ]
        return OptionsSnapshot(
            symbol=symbol, as_of=dt.date.today(), expiries=(near, far),
            quotes=tuple(quotes), put_call_oi=1.0, put_call_volume=1.0,
            max_pain=100.0, max_pain_expiry=near, atm_iv=0.4,
        )

    def get_intraday(self, symbol, days=60):
        raise DataNotFoundError("no intraday in tests", provider=self.name)


# The real universe is ~95 instruments; pricing all of them through the fake
# provider would write tens of thousands of rows per test. The recap page is
# exercised against a handful that still spans every region, asset class and
# quote kind the renderer branches on.
SMOKE_UNIVERSE = [
    Instrument("^N225", "Nikkei 225", Region.APAC, AssetClass.EQUITIES),
    Instrument("^FTSE", "FTSE 100", Region.UK, AssetClass.EQUITIES),
    Instrument("^GSPC", "S&P 500", Region.US, AssetClass.EQUITIES),
    Instrument("^TNX", "US 10-year", Region.US, AssetClass.FIXED_INCOME, QuoteKind.YIELD),
    Instrument("EURUSD=X", "EUR/USD", Region.EUROPE, AssetClass.CURRENCIES, QuoteKind.FX),
    Instrument("GC=F", "Gold", Region.GLOBAL, AssetClass.COMMODITIES),
]


@pytest.fixture()
def small_universe(monkeypatch):
    """Shrink the recap universe so page rendering stays fast."""
    monkeypatch.setattr(
        "market_intel.services.recap.instruments_for",
        lambda region=None, asset_class=None: [
            instrument
            for instrument in SMOKE_UNIVERSE
            if (region is None or instrument.region is region)
            and (asset_class is None or instrument.asset_class is asset_class)
        ],
    )


@pytest.fixture()
def fake_services(monkeypatch, small_universe):
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(
        _env_file=None, mkr_history_days=420, mkr_monte_carlo_paths=100
    )
    cache = ApiCache(db)
    market_data = MarketDataService(db, FakeMarket(), cache, settings)
    recap = RecapService(db, market_data, settings)
    news = NewsService(db, FakeNews(), cache, settings, market_data)
    services = ui.context.AppServices(
        settings=settings,
        db=db,
        market_data=market_data,
        news=news,
        calendar=CalendarService(db, FakeEvents(), cache, settings, market_data),
        etf=EtfService(db, FakeEtf(), cache, settings, market_data),
        scanner=ScannerService(market_data),
        recap=recap,
        mkr=MkrService(db, market_data, settings, cache, FakeFundamentalsFeed()),
        reports=ReportPipeline(recap, news),
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


def test_mkr_page_runs_the_full_analysis(fake_services) -> None:
    """The populated page is the interesting one: every framework must render."""
    at = AppTest.from_file("ui/pages/mkr.py", default_timeout=120)
    at.run()
    assert not at.exception, at.exception

    at.text_input[0].set_value("NVDA")
    at.button[0].click().run()

    assert not at.exception, at.exception
    # Fourteen scorecard rows plus the levels, options and Monte Carlo tables.
    assert at.dataframe
    rendered = " ".join(block.value for block in at.markdown)
    assert "SCORECARD TABLE" in rendered
