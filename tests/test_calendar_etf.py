"""Tests for the calendar and ETF modules."""

import datetime as dt

import pytest

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.models import CalendarEvent, EtfHoldingItem, PriceBar, SecurityInfo
from market_intel.providers.base import (
    EtfHoldingsProvider,
    EventProvider,
    MarketDataProvider,
)
from market_intel.services import MarketDataService
from market_intel.services.calendar import CalendarService
from market_intel.services.etf import EtfService

UTC = dt.timezone.utc


class FakeMarketProvider(MarketDataProvider):
    name = "fake"

    def get_security_info(self, symbol: str) -> SecurityInfo:
        return SecurityInfo(symbol=symbol)

    def get_daily_bars(self, symbol, start, end) -> list[PriceBar]:
        raise NotImplementedError


class FakeEventProvider(EventProvider):
    name = "fake"

    def __init__(self, events: dict[str, list[CalendarEvent]]) -> None:
        self.events = events
        self.calls: list[str] = []

    def get_earnings_events(self, symbol: str) -> list[CalendarEvent]:
        self.calls.append(symbol)
        return self.events.get(symbol, [])


class FakeEtfProvider(EtfHoldingsProvider):
    name = "fake"

    def __init__(self, holdings: dict[str, list[EtfHoldingItem]], fail: bool = False):
        self.holdings = holdings
        self.fail = fail

    def get_etf_holdings(self, symbol: str) -> list[EtfHoldingItem]:
        if self.fail:
            raise ProviderError("down", provider=self.name)
        if symbol not in self.holdings:
            raise DataNotFoundError("not an etf", provider=self.name)
        return self.holdings[symbol]


@pytest.fixture()
def db() -> Database:
    database = Database("sqlite:///:memory:")
    database.create_all()
    return database


def _wiring(db: Database):
    settings = Settings(_env_file=None)
    cache = ApiCache(db)
    market_data = MarketDataService(db, FakeMarketProvider(), cache, settings)
    return settings, cache, market_data


def _earnings(symbol: str, days_ahead: int, consensus=None, actual=None) -> CalendarEvent:
    """Build an earnings event ``days_ahead`` days from now.

    Deliberately relative to the current date: an absolute date here silently
    ages out of the ``get_upcoming`` window and the test starts failing on a
    calendar boundary rather than on a code change.
    """
    scheduled = dt.datetime.now(UTC).replace(
        hour=21, minute=0, second=0, microsecond=0
    ) + dt.timedelta(days=days_ahead)
    return CalendarEvent(
        event_type="earnings",
        title=f"{symbol} earnings",
        scheduled_at=scheduled,
        symbol=symbol,
        consensus=consensus,
        actual=actual,
        source="fake",
    )


class TestCalendarService:
    def _service(self, db: Database, provider: FakeEventProvider) -> CalendarService:
        settings, cache, market_data = _wiring(db)
        return CalendarService(db, provider, cache, settings, market_data)

    def test_refresh_stores_upcoming_events(self, db: Database) -> None:
        provider = FakeEventProvider({"NVDA": [_earnings("NVDA", 15, consensus=1.1)]})
        service = self._service(db, provider)

        assert service.refresh_earnings(["NVDA"]) == 1
        upcoming = service.get_upcoming(days_ahead=30)
        assert len(upcoming) == 1
        assert upcoming[0]["symbol"] == "NVDA"
        assert upcoming[0]["consensus"] == 1.1

    def test_refetch_updates_in_place(self, db: Database) -> None:
        from market_intel.database.repositories.events import EventRepository

        provider = FakeEventProvider({"NVDA": [_earnings("NVDA", 15, consensus=1.1)]})
        service = self._service(db, provider)
        service.refresh_earnings(["NVDA"])

        # Re-ingest the same event after the print: it now carries the actual.
        # (Repository level — the service's TTL marker would throttle a refetch.)
        with db.session() as session:
            from market_intel.database.repositories.securities import SecurityRepository

            security = SecurityRepository(session).get_by_symbol("NVDA")
            EventRepository(session).upsert(
                _earnings("NVDA", 15, consensus=1.1, actual=1.3),
                security_id=security.id,
            )

        upcoming = service.get_upcoming(days_ahead=30)
        assert len(upcoming) == 1  # updated, not duplicated
        assert upcoming[0]["actual"] == 1.3

    def test_refresh_throttled_by_marker(self, db: Database) -> None:
        provider = FakeEventProvider({"NVDA": [_earnings("NVDA", 15)]})
        service = self._service(db, provider)
        service.refresh_earnings(["NVDA"])
        service.refresh_earnings(["NVDA"])
        assert provider.calls == ["NVDA"]

    def test_manual_macro_event_and_window(self, db: Database) -> None:
        service = self._service(db, FakeEventProvider({}))
        now = dt.datetime.now(UTC)
        service.add_manual_event("CPI print", now + dt.timedelta(days=3), notes="core 0.3% exp")
        service.add_manual_event("Too far out", now + dt.timedelta(days=60))

        upcoming = service.get_upcoming(days_ahead=14)
        assert [e["title"] for e in upcoming] == ["CPI print"]
        assert upcoming[0]["type"] == "macro"


class TestEtfService:
    def _service(self, db: Database, provider: FakeEtfProvider) -> EtfService:
        settings, cache, market_data = _wiring(db)
        return EtfService(db, provider, cache, settings, market_data)

    def test_refresh_and_get_holdings(self, db: Database) -> None:
        provider = FakeEtfProvider(
            {
                "SMH": [
                    EtfHoldingItem("NVDA", weight=0.20, holding_name="NVIDIA"),
                    EtfHoldingItem("TSM", weight=0.12),
                ]
            }
        )
        service = self._service(db, provider)

        assert service.refresh_holdings("SMH") == 2
        frame = service.get_holdings("SMH")
        assert list(frame["symbol"]) == ["NVDA", "TSM"]
        assert frame.iloc[0]["weight"] == 0.20

    def test_exposure_across_etfs(self, db: Database) -> None:
        provider = FakeEtfProvider(
            {
                "SMH": [EtfHoldingItem("NVDA", weight=0.20)],
                "QQQ": [EtfHoldingItem("NVDA", weight=0.08), EtfHoldingItem("MSFT", weight=0.09)],
            }
        )
        service = self._service(db, provider)
        service.refresh_holdings("SMH")
        service.refresh_holdings("QQQ")

        exposure = service.get_exposure("nvda")
        assert {e["etf"]: e["weight"] for e in exposure} == {"SMH": 0.20, "QQQ": 0.08}
        assert service.get_exposure("ZZZZ") == []

    def test_provider_failure_keeps_stored_snapshot(self, db: Database) -> None:
        provider = FakeEtfProvider({"SMH": [EtfHoldingItem("NVDA", weight=0.2)]})
        service = self._service(db, provider)
        service.refresh_holdings("SMH")

        failing = self._service(db, FakeEtfProvider({}, fail=True))
        assert failing.refresh_holdings("SMH") == 0  # no raise
        assert not failing.get_holdings("SMH").empty

    def test_provider_failure_without_snapshot_raises(self, db: Database) -> None:
        service = self._service(db, FakeEtfProvider({}, fail=True))
        with pytest.raises(ProviderError):
            service.refresh_holdings("SMH")

    def test_refresh_many_reports_per_symbol_outcomes(self, db: Database) -> None:
        provider = FakeEtfProvider({"SMH": [EtfHoldingItem("NVDA", weight=0.2)]})
        service = self._service(db, provider)

        outcomes = service.refresh_many(["SMH", "NOPE"])
        assert outcomes["SMH"] == "fetched"
        assert "not an etf" in outcomes["NOPE"]  # error captured, no raise

        # Second pass: TTL marker means the fetch is skipped.
        assert service.refresh_many(["SMH"]) == {"SMH": "cached"}

    def test_overlap_between_two_etfs(self, db: Database) -> None:
        provider = FakeEtfProvider(
            {
                "SMH": [
                    EtfHoldingItem("NVDA", weight=0.20, holding_name="NVIDIA"),
                    EtfHoldingItem("TSM", weight=0.12),
                ],
                "SOXX": [
                    EtfHoldingItem("NVDA", weight=0.08),
                    EtfHoldingItem("AVGO", weight=0.09),
                ],
            }
        )
        service = self._service(db, provider)
        service.refresh_holdings("SMH")
        service.refresh_holdings("SOXX")

        overlap = service.get_overlap("SMH", "SOXX")
        assert list(overlap["symbol"]) == ["NVDA"]
        assert overlap.iloc[0]["overlap_weight"] == pytest.approx(0.08)
        assert overlap.iloc[0]["weight_a"] == pytest.approx(0.20)

        # No snapshot on one side -> empty frame, no raise.
        assert service.get_overlap("SMH", "QQQ").empty


class TestEtfCatalog:
    def test_catalog_symbols_unique_and_lookupable(self) -> None:
        from market_intel.etf_catalog import all_symbols, etfs_for_theme, lookup, themes

        symbols = all_symbols()
        assert len(symbols) == len(set(symbols))
        assert "SPY" in symbols
        assert lookup("smh").name == "VanEck Semiconductor"
        assert lookup("ZZZZ") is None
        assert etfs_for_theme("US sectors")
        assert etfs_for_theme("no such theme") == []
        assert all(etfs_for_theme(theme) for theme in themes())
