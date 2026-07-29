"""Tests for the database layer against an in-memory SQLite database."""

import datetime as dt

import pytest

from market_intel.database import Database
from market_intel.database.repositories import PriceRepository, SecurityRepository
from market_intel.models import PriceBar, SecurityInfo


@pytest.fixture()
def db() -> Database:
    database = Database("sqlite:///:memory:")
    database.create_all()
    return database


def _bar(day: int, close: float, source: str = "test") -> PriceBar:
    return PriceBar(
        date=dt.date(2026, 7, day),
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        adj_close=close,
        volume=1_000_000,
        source=source,
    )


class TestSecurityRepository:
    def test_upsert_inserts_and_normalises_symbol(self, db: Database) -> None:
        with db.session() as session:
            repo = SecurityRepository(session)
            security = repo.upsert(SecurityInfo(symbol=" nvda ", name="NVIDIA Corp"))
            assert security.symbol == "NVDA"
            assert security.id is not None

    def test_upsert_merges_only_non_null_fields(self, db: Database) -> None:
        with db.session() as session:
            repo = SecurityRepository(session)
            repo.upsert(SecurityInfo(symbol="NVDA", name="NVIDIA Corp", sector="Tech"))
            # Second upsert without name/sector must not wipe them.
            repo.upsert(SecurityInfo(symbol="NVDA", industry="Semiconductors"))

        with db.session() as session:
            security = SecurityRepository(session).get_by_symbol("nvda")
            assert security is not None
            assert security.name == "NVIDIA Corp"
            assert security.sector == "Tech"
            assert security.industry == "Semiconductors"

    def test_get_by_symbol_missing_returns_none(self, db: Database) -> None:
        with db.session() as session:
            assert SecurityRepository(session).get_by_symbol("ZZZZ") is None

    def test_list_active(self, db: Database) -> None:
        with db.session() as session:
            repo = SecurityRepository(session)
            repo.upsert(SecurityInfo(symbol="MSFT"))
            repo.upsert(SecurityInfo(symbol="AAPL"))
            symbols = [s.symbol for s in repo.list_active()]
            assert symbols == ["AAPL", "MSFT"]


class TestPriceRepository:
    @pytest.fixture()
    def security_id(self, db: Database) -> int:
        with db.session() as session:
            return SecurityRepository(session).upsert(SecurityInfo(symbol="SPY")).id

    def test_upsert_and_read_back(self, db: Database, security_id: int) -> None:
        bars = [_bar(6, 100.0), _bar(7, 101.5), _bar(8, 99.75)]
        with db.session() as session:
            written = PriceRepository(session).upsert_bars(security_id, bars)
            assert written == 3

        with db.session() as session:
            history = PriceRepository(session).get_history(security_id)
            assert [b.close for b in history] == [100.0, 101.5, 99.75]
            assert history[0].date == dt.date(2026, 7, 6)

    def test_upsert_is_idempotent_and_updates(self, db: Database, security_id: int) -> None:
        with db.session() as session:
            PriceRepository(session).upsert_bars(security_id, [_bar(6, 100.0)])
        # Re-ingest the same date with a revised close (e.g. adjusted data).
        with db.session() as session:
            PriceRepository(session).upsert_bars(
                security_id, [_bar(6, 100.25, source="revised")]
            )

        with db.session() as session:
            history = PriceRepository(session).get_history(security_id)
            assert len(history) == 1
            assert history[0].close == 100.25
            assert history[0].source == "revised"

    def test_get_history_date_bounds(self, db: Database, security_id: int) -> None:
        with db.session() as session:
            PriceRepository(session).upsert_bars(
                security_id, [_bar(d, 100.0 + d) for d in (1, 2, 3, 6, 7)]
            )

        with db.session() as session:
            window = PriceRepository(session).get_history(
                security_id, start=dt.date(2026, 7, 2), end=dt.date(2026, 7, 6)
            )
            assert [b.date.day for b in window] == [2, 3, 6]

    def test_latest_date(self, db: Database, security_id: int) -> None:
        with db.session() as session:
            repo = PriceRepository(session)
            assert repo.latest_date(security_id) is None
            repo.upsert_bars(security_id, [_bar(6, 100.0), _bar(8, 101.0)])
            assert repo.latest_date(security_id) == dt.date(2026, 7, 8)

    def test_empty_upsert_is_noop(self, db: Database, security_id: int) -> None:
        with db.session() as session:
            assert PriceRepository(session).upsert_bars(security_id, []) == 0

    def test_session_rolls_back_on_error(self, db: Database) -> None:
        with pytest.raises(RuntimeError):
            with db.session() as session:
                SecurityRepository(session).upsert(SecurityInfo(symbol="TSLA"))
                raise RuntimeError("boom")

        with db.session() as session:
            assert SecurityRepository(session).get_by_symbol("TSLA") is None
