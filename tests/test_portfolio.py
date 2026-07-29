"""Tests for watchlist, theme and journal services."""

import datetime as dt

import pytest

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider
from market_intel.services import MarketDataService
from market_intel.services.portfolio import JournalService, ThemeService, WatchlistService


class FakeProvider(MarketDataProvider):
    """Deterministic prices: symbol-dependent constant daily drift."""

    name = "fake"
    DRIFT = {"NVDA": 1.01, "AMD": 0.995, "MSFT": 1.002}

    def get_security_info(self, symbol: str) -> SecurityInfo:
        return SecurityInfo(symbol=symbol)

    def get_daily_bars(self, symbol, start, end) -> list[PriceBar]:
        drift = self.DRIFT.get(symbol, 1.0)
        bars, price = [], 100.0
        for i in range((end - start).days + 1):
            price *= drift
            bars.append(PriceBar(date=start + dt.timedelta(days=i), close=price, source="fake"))
        return bars


@pytest.fixture()
def db() -> Database:
    database = Database("sqlite:///:memory:")
    database.create_all()
    return database


@pytest.fixture()
def market_data(db: Database) -> MarketDataService:
    return MarketDataService(db, FakeProvider(), ApiCache(db), Settings(_env_file=None))


class TestWatchlistService:
    def test_add_list_remove(self, db, market_data) -> None:
        service = WatchlistService(db, market_data)
        service.add_symbol("Core", "nvda")
        service.add_symbol("Core", "MSFT")
        service.add_symbol("Core", "NVDA")  # duplicate is a no-op

        assert service.get_symbols("Core") == ["MSFT", "NVDA"]
        assert service.list_watchlists() == [{"name": "Core", "symbols": ["MSFT", "NVDA"]}]

        assert service.remove_symbol("Core", "MSFT") is True
        assert service.remove_symbol("Core", "ZZZZ") is False
        assert service.get_symbols("Core") == ["NVDA"]


class TestThemeService:
    def test_membership_and_performance(self, db, market_data) -> None:
        service = ThemeService(db, market_data)
        service.create_theme("AI", "Artificial intelligence beneficiaries")
        service.add_member("AI", "NVDA", weight=0.5)
        service.add_member("AI", "AMD")

        themes = service.list_themes()
        assert themes[0]["name"] == "AI"
        assert {m["symbol"] for m in themes[0]["members"]} == {"NVDA", "AMD"}

        perf = service.performance("AI", days=30)
        assert [row["symbol"] for row in perf] == ["NVDA", "AMD"]  # sorted by return
        assert perf[0]["return"] > 0 > perf[1]["return"]


class TestJournalService:
    def test_idea_open_close_lifecycle(self, db, market_data) -> None:
        service = JournalService(db, market_data)
        entry_id = service.log_idea(
            "Underreaction to guidance raise",
            symbol="NVDA",
            direction="long",
            target_price=200.0,
        )

        assert service.open_position(entry_id, entry_price=150.0)
        assert service.close_position(entry_id, exit_price=165.0, review="Thesis played out")

        (entry,) = service.list_entries()
        assert entry["symbol"] == "NVDA"
        assert entry["status"] == "closed"
        assert entry["pnl_pct"] == pytest.approx(0.10)
        assert entry["review"] == "Thesis played out"

    def test_short_pnl_sign(self, db, market_data) -> None:
        service = JournalService(db, market_data)
        entry_id = service.log_idea("Overreaction fade", symbol="AMD", direction="short")
        service.open_position(entry_id, entry_price=100.0)
        service.close_position(entry_id, exit_price=90.0)
        (entry,) = service.list_entries(status="closed")
        assert entry["pnl_pct"] == pytest.approx(0.10)

    def test_update_missing_entry_returns_false(self, db, market_data) -> None:
        service = JournalService(db, market_data)
        assert service.open_position(999, entry_price=1.0) is False
