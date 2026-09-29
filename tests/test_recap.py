"""Tests for the global market recap: analysis, service, and the new providers."""

import datetime as dt

import pandas as pd
import pytest

from market_intel.analysis.recap import (
    SessionState,
    block_stats,
    compute_move,
    group_moves,
    rank_by_magnitude,
    rank_rates_by_bp,
    session_label,
    session_state,
    top_gainers,
    top_losers,
)
from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    ProviderError,
    RateLimitError,
)
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.alphavantage_provider import AlphaVantageProvider
from market_intel.providers.base import MarketDataProvider
from market_intel.providers.chain import ChainedMarketDataProvider
from market_intel.providers.fred_provider import FredProvider, _parse_observation
from market_intel.services.market_data import MarketDataService
from market_intel.services.recap import RecapService, format_snapshot
from market_intel.universe import (
    RECAP_UNIVERSE,
    AssetClass,
    Instrument,
    QuoteKind,
    Region,
    find,
    instruments_for,
    symbols_for,
)

UTC = dt.timezone.utc

SPX = Instrument("^GSPC", "S&P 500", Region.US, AssetClass.EQUITIES)
UST10 = Instrument(
    "^TNX", "US 10-year", Region.US, AssetClass.FIXED_INCOME, QuoteKind.YIELD
)
EURUSD = Instrument(
    "EURUSD=X", "EUR/USD", Region.EUROPE, AssetClass.CURRENCIES, QuoteKind.FX
)
NIKKEI = Instrument("^N225", "Nikkei 225", Region.APAC, AssetClass.EQUITIES)


def _series(values: list[float], end: dt.date | None = None) -> pd.Series:
    """Build a date-indexed close series ending at ``end`` (default today)."""
    end = end or dt.date.today()
    index = [end - dt.timedelta(days=len(values) - 1 - i) for i in range(len(values))]
    return pd.Series(values, index=pd.Index(index, name="date"))


# --- Universe -----------------------------------------------------------------


class TestUniverse:
    def test_every_instrument_has_a_unique_symbol(self) -> None:
        symbols = [instrument.symbol for instrument in RECAP_UNIVERSE]
        assert len(symbols) == len(set(symbols))

    def test_all_four_asset_classes_and_regions_are_populated(self) -> None:
        for region in Region:
            assert instruments_for(region), f"{region} has no instruments"
        for asset_class in AssetClass:
            assert instruments_for(asset_class=asset_class)

    def test_filters_compose(self) -> None:
        us_fx = instruments_for(Region.US, AssetClass.CURRENCIES)
        assert us_fx
        assert all(i.region is Region.US for i in us_fx)
        assert all(i.asset_class is AssetClass.CURRENCIES for i in us_fx)

    def test_bond_proxies_are_flagged_and_explained(self) -> None:
        proxies = [i for i in RECAP_UNIVERSE if i.is_proxy]
        assert proxies, "expected ETF stand-ins for non-US rates"
        # A proxy that does not say it is one would mislead the reader.
        assert all(i.note for i in proxies)
        assert all(i.inverse_yield for i in proxies)

    def test_find_is_case_insensitive(self) -> None:
        assert find("^gspc") is find("^GSPC") is not None
        assert find("NOT_A_TICKER") is None

    def test_symbols_for_matches_instruments_for(self) -> None:
        assert symbols_for(Region.UK) == [i.symbol for i in instruments_for(Region.UK)]


# --- Session clock ------------------------------------------------------------


class TestSessionState:
    def test_apac_open_us_closed_during_asian_morning(self) -> None:
        # Friday 03:25 UTC - Tokyo/HK mid-session, New York asleep.
        now = dt.datetime(2026, 8, 21, 3, 25)
        assert session_state(Region.APAC, now) is SessionState.OPEN
        assert session_state(Region.US, now) is SessionState.CLOSED
        assert session_state(Region.UK, now) is SessionState.CLOSED

    def test_us_open_apac_closed_during_new_york_afternoon(self) -> None:
        now = dt.datetime(2026, 8, 20, 18, 0)
        assert session_state(Region.US, now) is SessionState.OPEN
        assert session_state(Region.APAC, now) is SessionState.CLOSED

    def test_london_open_mid_european_morning(self) -> None:
        now = dt.datetime(2026, 8, 20, 9, 0)
        assert session_state(Region.UK, now) is SessionState.OPEN
        assert session_state(Region.EUROPE, now) is SessionState.OPEN

    def test_weekend_beats_session_hours(self) -> None:
        saturday = dt.datetime(2026, 8, 22, 3, 0)
        assert session_state(Region.APAC, saturday) is SessionState.WEEKEND
        sunday = dt.datetime(2026, 8, 23, 18, 0)
        assert session_state(Region.US, sunday) is SessionState.WEEKEND

    def test_aware_datetimes_are_converted_not_rejected(self) -> None:
        # Same instant, expressed in Singapore time.
        sgt = dt.timezone(dt.timedelta(hours=8))
        aware = dt.datetime(2026, 8, 21, 11, 25, tzinfo=sgt)
        assert session_state(Region.APAC, aware) is SessionState.OPEN

    def test_labels_distinguish_live_from_last_close(self) -> None:
        assert "Live" in session_label(Region.APAC, SessionState.OPEN)
        assert "last completed" in session_label(Region.US, SessionState.CLOSED)
        assert "Friday" in session_label(Region.US, SessionState.WEEKEND)


# --- Move computation ---------------------------------------------------------


class TestComputeMove:
    def test_percentage_move_for_an_index(self) -> None:
        move = compute_move(SPX, _series([100.0, 110.0]))
        assert move is not None
        assert move.last == 110.0
        assert move.change == pytest.approx(10.0)
        assert move.pct_change == pytest.approx(10.0)
        assert move.bp_change is None
        assert move.display_change == "+10.00%"

    def test_yield_move_is_expressed_in_basis_points(self) -> None:
        # 4.66% -> 4.70% is +4bp, not +0.86%.
        move = compute_move(UST10, _series([4.66, 4.70]))
        assert move is not None
        assert move.bp_change == pytest.approx(4.0)
        assert move.display_change == "+4.0 bp"

    def test_returns_none_without_two_observations(self) -> None:
        assert compute_move(SPX, _series([100.0])) is None
        assert compute_move(SPX, pd.Series(dtype=float)) is None

    def test_returns_none_when_previous_close_is_zero(self) -> None:
        assert compute_move(SPX, _series([0.0, 5.0])) is None

    def test_non_numeric_values_are_dropped(self) -> None:
        series = pd.Series(
            [100.0, None, 110.0],
            index=pd.Index(
                [dt.date(2026, 8, 18), dt.date(2026, 8, 19), dt.date(2026, 8, 20)]
            ),
        )
        move = compute_move(SPX, series)
        assert move is not None
        assert move.observations == 2
        assert move.pct_change == pytest.approx(10.0)

    def test_window_returns_are_none_when_history_is_short(self) -> None:
        move = compute_move(SPX, _series([100.0, 101.0, 102.0]))
        assert move is not None
        assert move.ret_5d is None
        assert move.ret_1m is None

    def test_window_returns_populate_with_enough_history(self) -> None:
        move = compute_move(SPX, _series([100.0] * 25 + [110.0]))
        assert move is not None
        assert move.ret_5d == pytest.approx(10.0)
        assert move.ret_1m == pytest.approx(10.0)

    def test_ytd_uses_the_first_observation_of_the_current_year(self) -> None:
        index = [dt.date(2025, 12, 30), dt.date(2026, 1, 2), dt.date(2026, 8, 20)]
        series = pd.Series([50.0, 100.0, 150.0], index=pd.Index(index))
        move = compute_move(SPX, series)
        assert move is not None
        # Measured from the January print, not from the December one.
        assert move.ret_ytd == pytest.approx(50.0)

    def test_staleness_tolerates_a_long_weekend(self) -> None:
        move = compute_move(SPX, _series([100.0, 101.0], end=dt.date(2026, 8, 20)))
        assert move is not None
        assert not move.is_stale(dt.date(2026, 8, 23))
        assert move.is_stale(dt.date(2026, 9, 1))


# --- Ranking and grouping -----------------------------------------------------


class TestRanking:
    @pytest.fixture()
    def moves(self):
        # Magnitudes are kept clearly distinct: equal-and-opposite moves would
        # tie, and the ordering would then turn on floating-point noise.
        return [
            compute_move(SPX, _series([100.0, 99.0])),  # -1.00%
            compute_move(NIKKEI, _series([100.0, 108.0])),  # +8.00%
            compute_move(EURUSD, _series([1.0, 1.02])),  # +2.00%
            compute_move(UST10, _series([4.0, 3.8])),  # -5.00%
        ]

    def test_rank_by_magnitude_ignores_direction(self, moves) -> None:
        # Ranked on absolute size, and rates are excluded: the -5% on the
        # 10-year is a rate move and belongs in the basis-point ranking.
        ranked = rank_by_magnitude(moves, limit=2)
        assert [m.instrument.symbol for m in ranked] == ["^N225", "EURUSD=X"]

    def test_gainers_and_losers_are_signed_and_ordered(self, moves) -> None:
        assert [m.instrument.symbol for m in top_gainers(moves)] == ["^N225", "EURUSD=X"]
        # ^TNX is a rate, so it is ranked in basis points, not among losers.
        assert [m.instrument.symbol for m in top_losers(moves)] == ["^GSPC"]

    def test_limit_is_respected(self, moves) -> None:
        assert len(rank_by_magnitude(moves, limit=1)) == 1

    def test_block_stats_counts_breadth(self, moves) -> None:
        stats = block_stats(moves)
        assert stats.count == 4
        assert stats.advancers == 2
        assert stats.decliners == 2
        assert stats.widest.instrument.symbol == "^N225"

    def test_block_stats_handles_an_empty_block(self) -> None:
        stats = block_stats([])
        assert stats.count == 0
        assert stats.widest is None

    def test_a_narrow_spread_does_not_dominate_the_movers_list(self) -> None:
        """A 4bp move on a near-zero spread must not outrank a 7% VIX spike.

        The 10y-2y curve going 0.46 -> 0.50 is +8.7% in percentage terms. Ranking
        that against equities by percent would put a trivial rates move at the
        top of the board, which is what this guards against.
        """
        curve = Instrument(
            "FRED:10Y2Y", "10y-2y curve", Region.US, AssetClass.FIXED_INCOME,
            QuoteKind.YIELD,
        )
        vix = Instrument("^VIX", "VIX", Region.US, AssetClass.EQUITIES)
        moves = [
            compute_move(curve, _series([0.46, 0.50])),
            compute_move(vix, _series([14.89, 16.01])),
        ]

        ranked = rank_by_magnitude(moves, limit=5)

        assert [m.instrument.symbol for m in ranked] == ["^VIX"]

    def test_rates_are_ranked_separately_in_basis_points(self) -> None:
        curve = Instrument(
            "FRED:10Y2Y", "10y-2y curve", Region.US, AssetClass.FIXED_INCOME,
            QuoteKind.YIELD,
        )
        moves = [
            compute_move(curve, _series([0.46, 0.50])),  # +4bp
            compute_move(UST10, _series([4.40, 4.70])),  # +30bp
        ]

        ranked = rank_rates_by_bp(moves)

        assert [m.instrument.symbol for m in ranked] == ["^TNX", "FRED:10Y2Y"]

    def test_block_average_excludes_rates(self) -> None:
        curve = Instrument(
            "FRED:10Y2Y", "10y-2y curve", Region.US, AssetClass.FIXED_INCOME,
            QuoteKind.YIELD,
        )
        moves = [
            compute_move(SPX, _series([100.0, 101.0])),  # +1%
            compute_move(curve, _series([0.46, 0.50])),  # +8.7% but only 4bp
        ]

        stats = block_stats(moves)

        assert stats.average_pct == pytest.approx(1.0)
        assert stats.advancers == 2  # breadth still counts everything
        assert stats.widest.instrument.symbol == "^GSPC"

    def test_gainers_and_losers_exclude_rates(self) -> None:
        moves = [
            compute_move(SPX, _series([100.0, 101.0])),
            compute_move(UST10, _series([4.0, 4.5])),
        ]
        assert all(not m.is_rate for m in top_gainers(moves))

    def test_group_moves_buckets_by_region_then_asset_class(self, moves) -> None:
        grouped = group_moves(moves)
        assert set(grouped) == {Region.US, Region.APAC, Region.EUROPE}
        assert AssetClass.FIXED_INCOME in grouped[Region.US]
        assert len(grouped[Region.US][AssetClass.EQUITIES]) == 1


# --- Providers ----------------------------------------------------------------


class RecordingProvider(MarketDataProvider):
    """A provider that returns canned bars and counts how often it was asked."""

    def __init__(self, name: str, close: float = 100.0) -> None:
        self.name = name
        self._close = close
        self.calls = 0

    def get_security_info(self, symbol):
        self.calls += 1
        return SecurityInfo(symbol=symbol, name=f"{symbol} via {self.name}")

    def get_daily_bars(self, symbol, start, end):
        self.calls += 1
        return [
            PriceBar(date=start, close=self._close, source=self.name),
            PriceBar(date=end, close=self._close * 1.01, source=self.name),
        ]


class FailingProvider(MarketDataProvider):
    """A provider that always raises the error it was constructed with."""

    def __init__(self, name: str, error: Exception) -> None:
        self.name = name
        self._error = error
        self.calls = 0

    def get_security_info(self, symbol):
        self.calls += 1
        raise self._error

    def get_daily_bars(self, symbol, start, end):
        self.calls += 1
        raise self._error


class TestChainedProvider:
    def test_primary_wins_and_fallback_is_never_called(self) -> None:
        primary = RecordingProvider("primary")
        backup = RecordingProvider("backup")
        chain = ChainedMarketDataProvider([primary, backup])

        bars = chain.get_daily_bars("^GSPC", dt.date(2026, 8, 1), dt.date(2026, 8, 20))

        assert bars[0].source == "primary"
        assert backup.calls == 0

    def test_outage_in_primary_falls_through_to_backup(self) -> None:
        primary = FailingProvider("primary", ProviderError("down", provider="primary"))
        backup = RecordingProvider("backup")
        chain = ChainedMarketDataProvider([primary, backup])

        bars = chain.get_daily_bars("^GSPC", dt.date(2026, 8, 1), dt.date(2026, 8, 20))

        assert bars[0].source == "backup"
        assert primary.calls == 1

    def test_uncovered_symbol_falls_through_quietly(self) -> None:
        primary = FailingProvider(
            "primary", DataNotFoundError("no cover", provider="primary")
        )
        backup = RecordingProvider("backup")
        chain = ChainedMarketDataProvider([primary, backup])

        assert chain.get_daily_bars("X", dt.date(2026, 8, 1), dt.date(2026, 8, 2))

    def test_empty_result_is_treated_as_no_coverage(self) -> None:
        class EmptyProvider(MarketDataProvider):
            name = "empty"

            def get_security_info(self, symbol):
                return SecurityInfo(symbol=symbol)

            def get_daily_bars(self, symbol, start, end):
                return []

        backup = RecordingProvider("backup")
        chain = ChainedMarketDataProvider([EmptyProvider(), backup])

        bars = chain.get_daily_bars("^GSPC", dt.date(2026, 8, 1), dt.date(2026, 8, 2))
        assert bars[0].source == "backup"

    def test_all_outages_raise_provider_error(self) -> None:
        chain = ChainedMarketDataProvider(
            [
                FailingProvider("a", ProviderError("down", provider="a")),
                FailingProvider("b", ProviderError("down", provider="b")),
            ]
        )
        with pytest.raises(ProviderError):
            chain.get_daily_bars("^GSPC", dt.date(2026, 8, 1), dt.date(2026, 8, 2))

    def test_all_not_found_raises_data_not_found(self) -> None:
        chain = ChainedMarketDataProvider(
            [
                FailingProvider("a", DataNotFoundError("no", provider="a")),
                FailingProvider("b", DataNotFoundError("no", provider="b")),
            ]
        )
        with pytest.raises(DataNotFoundError):
            chain.get_daily_bars("X", dt.date(2026, 8, 1), dt.date(2026, 8, 2))

    def test_a_broken_adapter_does_not_break_the_chain(self) -> None:
        primary = FailingProvider("primary", ValueError("adapter bug"))
        backup = RecordingProvider("backup")
        chain = ChainedMarketDataProvider([primary, backup])

        bars = chain.get_daily_bars("^GSPC", dt.date(2026, 8, 1), dt.date(2026, 8, 2))
        assert bars[0].source == "backup"

    def test_empty_chain_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            ChainedMarketDataProvider([])

    def test_name_records_the_composition(self) -> None:
        chain = ChainedMarketDataProvider(
            [RecordingProvider("yfinance"), RecordingProvider("fred")]
        )
        assert chain.name == "chain:yfinance+fred"


class TestFredProvider:
    def test_unmapped_symbol_is_not_found(self) -> None:
        provider = FredProvider()
        with pytest.raises(DataNotFoundError):
            provider.get_security_info("^N225")

    def test_mapped_symbol_returns_reference_data(self) -> None:
        info = FredProvider().get_security_info("^TNX")
        assert info.symbol == "^TNX"
        assert info.exchange == "FRED"

    def test_placeholder_observations_are_skipped(self) -> None:
        # FRED writes "." for holidays; a naive float() would raise.
        assert _parse_observation("2026-08-20", ".") is None
        assert _parse_observation("2026-08-20", "") is None
        assert _parse_observation("not-a-date", "4.70") is None
        assert _parse_observation("2026-08-20", "4.70") == (dt.date(2026, 8, 20), 4.70)


class TestAlphaVantageProvider:
    def test_without_a_key_every_symbol_is_uncovered(self) -> None:
        provider = AlphaVantageProvider(api_key=None)
        with pytest.raises(DataNotFoundError):
            provider.get_daily_bars("SPY", dt.date(2026, 8, 1), dt.date(2026, 8, 2))

    def test_indices_and_futures_are_rejected_without_spending_a_request(self) -> None:
        provider = AlphaVantageProvider(api_key="test-key")
        for symbol in ("^GSPC", "GC=F"):
            with pytest.raises(DataNotFoundError):
                provider.get_daily_bars(symbol, dt.date(2026, 8, 1), dt.date(2026, 8, 2))
        assert provider.calls_remaining == 25

    def test_budget_is_enforced_before_the_network(self) -> None:
        provider = AlphaVantageProvider(api_key="test-key", daily_budget=0)
        with pytest.raises(RateLimitError):
            provider.get_daily_bars("SPY", dt.date(2026, 8, 1), dt.date(2026, 8, 2))

    def test_budget_defaults_to_the_free_tier_allowance(self) -> None:
        assert AlphaVantageProvider(api_key="k").calls_remaining == 25


# --- Service ------------------------------------------------------------------


class FakeMarketProvider(MarketDataProvider):
    """Deterministic bars: a steady climb, so every move is positive."""

    name = "fake"

    def get_security_info(self, symbol):
        return SecurityInfo(symbol=symbol, name=symbol)

    def get_daily_bars(self, symbol, start, end):
        bars, price = [], 100.0
        day = start
        while day <= end:
            price *= 1.001
            bars.append(PriceBar(date=day, close=price, adj_close=price, source="fake"))
            day += dt.timedelta(days=1)
        return bars


SERVICE_UNIVERSE = [SPX, UST10, EURUSD, NIKKEI]


@pytest.fixture()
def recap_service(monkeypatch):
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(_env_file=None)
    market_data = MarketDataService(db, FakeMarketProvider(), ApiCache(db), settings)
    monkeypatch.setattr(
        "market_intel.services.recap.instruments_for",
        lambda region=None, asset_class=None: [
            i
            for i in SERVICE_UNIVERSE
            if (region is None or i.region is region)
            and (asset_class is None or i.asset_class is asset_class)
        ],
    )
    return RecapService(db, market_data, settings), db, settings


class TestRecapSnapshot:
    def test_snapshot_covers_every_region_in_the_universe(self, recap_service) -> None:
        service, _, _ = recap_service
        snapshot = service.build_snapshot(now=dt.datetime(2026, 8, 21, 3, 25, tzinfo=UTC))

        assert not snapshot.is_empty
        assert {block.region for block in snapshot.blocks} == {
            Region.US,
            Region.EUROPE,
            Region.APAC,
        }

    def test_session_state_is_stamped_per_region(self, recap_service) -> None:
        service, _, _ = recap_service
        snapshot = service.build_snapshot(now=dt.datetime(2026, 8, 21, 3, 25, tzinfo=UTC))

        assert snapshot.region(Region.APAC).state is SessionState.OPEN
        assert snapshot.region(Region.US).state is SessionState.CLOSED

    def test_progress_callback_reports_every_instrument(self, recap_service) -> None:
        service, _, _ = recap_service
        seen = []
        service.build_snapshot(progress=lambda done, total, symbol: seen.append(symbol))
        assert len(seen) == len(SERVICE_UNIVERSE)

    def test_a_dead_symbol_is_recorded_not_raised(self, monkeypatch, recap_service) -> None:
        service, _, _ = recap_service

        class PartlyBroken(FakeMarketProvider):
            def get_daily_bars(self, symbol, start, end):
                if symbol == "^GSPC":
                    raise ProviderError("boom", provider="fake")
                return super().get_daily_bars(symbol, start, end)

        monkeypatch.setattr(service._market_data, "_provider", PartlyBroken())
        snapshot = service.build_snapshot()

        assert "^GSPC" in snapshot.failures
        assert snapshot.moves, "the other instruments should still be priced"

    def test_biggest_movers_is_capped(self, recap_service) -> None:
        service, _, _ = recap_service
        snapshot = service.build_snapshot()
        assert len(snapshot.biggest_movers(2)) == 2


class TestFormatSnapshot:
    def test_payload_carries_levels_regions_and_session_status(self, recap_service) -> None:
        service, _, _ = recap_service
        snapshot = service.build_snapshot(now=dt.datetime(2026, 8, 21, 3, 25, tzinfo=UTC))

        payload = format_snapshot(snapshot)

        assert "MARKET SNAPSHOT" in payload
        assert "only figures you may quote" in payload
        assert "APAC" in payload and "US" in payload
        assert "Equities" in payload
        assert "Biggest price moves" in payload
        assert "Biggest rate and spread moves" in payload
        # The session status has to reach the model, or it will call a closed
        # market's last print "today's trading".
        assert "Live" in payload or "last completed" in payload

    def test_yield_lines_are_quoted_in_basis_points(self, recap_service) -> None:
        service, _, _ = recap_service
        snapshot = service.build_snapshot()
        payload = format_snapshot(snapshot)
        assert " bp" in payload

    def test_failures_are_declared_so_they_are_not_reported_as_moves(
        self, recap_service
    ) -> None:
        service, _, _ = recap_service
        snapshot = service.build_snapshot()
        object.__setattr__(snapshot, "failures", {"^HSI": "no data"})
        assert "do not mention as moves" in format_snapshot(snapshot)


class _StubResponse:
    stop_reason = "end_turn"
    model = "claude-opus-5"

    def __init__(self, text: str, searches: int = 0) -> None:
        blocks = [type("Text", (), {"type": "text", "text": text})()]
        blocks += [
            type("Tool", (), {"type": "server_tool_use", "name": "web_search"})()
            for _ in range(searches)
        ]
        self.content = blocks


class _StubMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _StubClient:
    def __init__(self, response) -> None:
        self.messages = _StubMessages(response)


class TestGenerateRecap:
    def test_missing_key_is_a_configuration_error(self, recap_service) -> None:
        service, _, _ = recap_service
        assert not service.available
        with pytest.raises(ConfigurationError):
            service.generate_recap(service.build_snapshot())

    def test_recap_is_written_and_archived(self, recap_service) -> None:
        service, db, settings = recap_service
        client = _StubClient(_StubResponse("## Market wrap\n\nRisk off.", searches=3))
        service = RecapService(db, service._market_data, settings, client=client)

        note = service.generate_recap(service.build_snapshot())

        assert note["content"].startswith("## Market wrap")
        assert note["note_type"] == "market_recap"
        assert note["searches"] == 3
        assert note["id"]

    def test_web_search_tool_is_offered_by_default(self, recap_service) -> None:
        service, db, settings = recap_service
        client = _StubClient(_StubResponse("wrap"))
        service = RecapService(db, service._market_data, settings, client=client)

        service.generate_recap(service.build_snapshot())

        tools = client.messages.kwargs["tools"]
        assert tools[0]["name"] == "web_search"
        assert tools[0]["max_uses"] == settings.recap_web_search_max_uses

    def test_web_search_can_be_disabled(self, recap_service) -> None:
        service, db, settings = recap_service
        client = _StubClient(_StubResponse("wrap"))
        service = RecapService(db, service._market_data, settings, client=client)

        service.generate_recap(service.build_snapshot(), use_web_search=False)

        assert not client.messages.kwargs["tools"]

    def test_focus_reaches_the_prompt(self, recap_service) -> None:
        service, db, settings = recap_service
        client = _StubClient(_StubResponse("wrap"))
        service = RecapService(db, service._market_data, settings, client=client)

        service.generate_recap(service.build_snapshot(), focus="rates and the yen")

        sent = client.messages.kwargs["messages"][0]["content"]
        assert "rates and the yen" in sent

    def test_empty_response_is_a_provider_error(self, recap_service) -> None:
        service, db, settings = recap_service
        client = _StubClient(_StubResponse("   "))
        service = RecapService(db, service._market_data, settings, client=client)

        with pytest.raises(ProviderError):
            service.generate_recap(service.build_snapshot())

    def test_refusal_is_a_provider_error(self, recap_service) -> None:
        service, db, settings = recap_service
        response = _StubResponse("partial")
        response.stop_reason = "refusal"
        service = RecapService(db, service._market_data, settings, client=_StubClient(response))

        with pytest.raises(ProviderError):
            service.generate_recap(service.build_snapshot())
