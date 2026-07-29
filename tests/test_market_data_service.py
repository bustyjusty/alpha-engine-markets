"""Tests for MarketDataService using a fake in-memory provider."""

import datetime as dt

import pytest

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories import PriceRepository, SecurityRepository
from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider
from market_intel.services import MarketDataService

D = dt.date  # brevity


class FakeProvider(MarketDataProvider):
    """Serves bars from a dict and records every call."""

    name = "fake"

    def __init__(self, bars: list[PriceBar] | None = None, fail: bool = False) -> None:
        self.bars = bars or []
        self.fail = fail
        self.bar_calls: list[tuple[str, dt.date, dt.date]] = []
        self.info_calls: list[str] = []

    def get_security_info(self, symbol: str) -> SecurityInfo:
        self.info_calls.append(symbol)
        if self.fail:
            raise ProviderError("provider down", provider=self.name)
        return SecurityInfo(symbol=symbol, name="Fake Corp")

    def get_daily_bars(self, symbol, start, end) -> list[PriceBar]:
        self.bar_calls.append((symbol, start, end))
        if self.fail:
            raise ProviderError("provider down", provider=self.name)
        found = [b for b in self.bars if start <= b.date <= end]
        if not found:
            raise DataNotFoundError("no bars", provider=self.name)
        return found


def _bar(day: int, close: float = 100.0) -> PriceBar:
    return PriceBar(date=D(2026, 7, day), close=close, source="fake")


@pytest.fixture()
def db() -> Database:
    database = Database("sqlite:///:memory:")
    database.create_all()
    return database


def _service(db: Database, provider: FakeProvider) -> MarketDataService:
    settings = Settings(_env_file=None)
    return MarketDataService(db, provider, ApiCache(db), settings)


def _seed_bars(db: Database, symbol: str, bars: list[PriceBar]) -> int:
    """Insert bars directly, bypassing the service."""
    with db.session() as session:
        security_id = SecurityRepository(session).upsert(SecurityInfo(symbol=symbol)).id
        PriceRepository(session).upsert_bars(security_id, bars)
    return security_id


class TestGetPriceHistory:
    def test_initial_fetch_stores_and_returns_frame(self, db: Database) -> None:
        provider = FakeProvider(bars=[_bar(6, 100.0), _bar(7, 101.0), _bar(8, 102.0)])
        service = _service(db, provider)

        frame = service.get_price_history("SPY", start=D(2026, 7, 6), end=D(2026, 7, 8))

        assert list(frame["close"]) == [100.0, 101.0, 102.0]
        assert provider.bar_calls == [("SPY", D(2026, 7, 6), D(2026, 7, 8))]
        # Bars were persisted.
        with db.session() as session:
            security = SecurityRepository(session).get_by_symbol("SPY")
            assert len(PriceRepository(session).get_history(security.id)) == 3

    def test_repeat_call_within_ttl_skips_provider(self, db: Database) -> None:
        provider = FakeProvider(bars=[_bar(6), _bar(7)])
        service = _service(db, provider)

        service.get_price_history("SPY", start=D(2026, 7, 6), end=D(2026, 7, 8))
        service.get_price_history("SPY", start=D(2026, 7, 6), end=D(2026, 7, 8))

        assert len(provider.bar_calls) == 1  # marker suppressed the second sync

    def test_incremental_fetch_only_missing_days(self, db: Database) -> None:
        _seed_bars(db, "SPY", [_bar(6), _bar(7)])
        provider = FakeProvider(bars=[_bar(8), _bar(9)])
        service = _service(db, provider)

        frame = service.get_price_history("SPY", start=D(2026, 7, 6), end=D(2026, 7, 9))

        assert provider.bar_calls == [("SPY", D(2026, 7, 8), D(2026, 7, 9))]
        assert len(frame) == 4

    def test_backfill_earlier_start(self, db: Database) -> None:
        _seed_bars(db, "SPY", [_bar(6), _bar(7)])
        provider = FakeProvider(bars=[_bar(1), _bar(2)])
        service = _service(db, provider)

        frame = service.get_price_history("SPY", start=D(2026, 7, 1), end=D(2026, 7, 7))

        assert provider.bar_calls == [("SPY", D(2026, 7, 1), D(2026, 7, 5))]
        assert [d.day for d in frame.index] == [1, 2, 6, 7]

    def test_provider_failure_serves_stored_data(self, db: Database) -> None:
        _seed_bars(db, "SPY", [_bar(6), _bar(7)])
        service = _service(db, FakeProvider(fail=True))

        frame = service.get_price_history("SPY", start=D(2026, 7, 6), end=D(2026, 7, 9))

        assert len(frame) == 2  # stored data, no exception

    def test_provider_failure_with_no_stored_data_raises(self, db: Database) -> None:
        service = _service(db, FakeProvider(fail=True))
        with pytest.raises(ProviderError):
            service.get_price_history("SPY", start=D(2026, 7, 6), end=D(2026, 7, 9))

    def test_no_new_bars_is_quiet(self, db: Database) -> None:
        """Forward window over a weekend returns nothing — not an error."""
        _seed_bars(db, "SPY", [_bar(3)])  # Friday
        provider = FakeProvider(bars=[])  # nothing for Sat/Sun
        service = _service(db, provider)

        frame = service.get_price_history("SPY", start=D(2026, 7, 3), end=D(2026, 7, 5))

        assert len(frame) == 1

    def test_empty_history_returns_empty_frame(self, db: Database) -> None:
        _seed_bars(db, "SPY", [_bar(6)])
        service = _service(db, FakeProvider(bars=[]))
        frame = service.get_price_history("SPY", start=D(2026, 1, 1), end=D(2026, 1, 31))
        assert frame.empty
        assert "close" in frame.columns


class TestEnsureSecurity:
    def test_info_cached_between_calls(self, db: Database) -> None:
        provider = FakeProvider()
        service = _service(db, provider)

        first = service.ensure_security("nvda")
        second = service.ensure_security("NVDA")

        assert first == second
        assert provider.info_calls == ["NVDA"]

    def test_provider_failure_falls_back_to_stored_record(self, db: Database) -> None:
        _seed_bars(db, "NVDA", [])
        service = _service(db, FakeProvider(fail=True))
        assert isinstance(service.ensure_security("NVDA"), int)

    def test_unknown_symbol_with_provider_down_raises(self, db: Database) -> None:
        service = _service(db, FakeProvider(fail=True))
        with pytest.raises(ProviderError):
            service.ensure_security("ZZZZ")
