"""Tests for the report pipeline, the narrative writer, and level formatting."""

import datetime as dt

import pandas as pd
import pytest

from market_intel.analysis.narrative import (
    compose_findings,
    describe_breadth,
    describe_region,
    find_divergences,
    headline_facts,
)
from market_intel.analysis.recap import (
    SessionState,
    compute_move,
    format_level,
    is_partial_session,
    level_format_for,
    rank_by_magnitude,
    top_gainers,
)
from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider
from market_intel.services.market_data import MarketDataService
from market_intel.services.recap import RecapService
from market_intel.services.report import (
    ReportPipeline,
    format_headlines,
    gather_headlines,
    interpret_prompt,
)
from market_intel.universe import AssetClass, Instrument, QuoteKind, Region

UTC = dt.timezone.utc

SPX = Instrument("^GSPC", "S&P 500", Region.US, AssetClass.EQUITIES)
RUT = Instrument("^RUT", "Russell 2000", Region.US, AssetClass.EQUITIES)
VIX = Instrument("^VIX", "VIX", Region.US, AssetClass.EQUITIES)
NIKKEI = Instrument("^N225", "Nikkei 225", Region.APAC, AssetClass.EQUITIES)
GOLD = Instrument("GC=F", "Gold", Region.GLOBAL, AssetClass.COMMODITIES)
CORN = Instrument("ZC=F", "Corn", Region.US, AssetClass.COMMODITIES)
BRENT = Instrument("BZ=F", "Brent crude", Region.EUROPE, AssetClass.COMMODITIES)
UST10 = Instrument("^TNX", "US 10-year", Region.US, AssetClass.FIXED_INCOME, QuoteKind.YIELD)
DXY = Instrument("DX-Y.NYB", "Dollar Index", Region.US, AssetClass.CURRENCIES, QuoteKind.FX)
USDJPY = Instrument("USDJPY=X", "USD/JPY", Region.APAC, AssetClass.CURRENCIES, QuoteKind.FX)


def _series(values, end=None):
    end = end or dt.date.today()
    index = [end - dt.timedelta(days=len(values) - 1 - i) for i in range(len(values))]
    return pd.Series(values, index=pd.Index(index, name="date"))


def _move(instrument, values, volumes=None):
    vol = _series(volumes) if volumes else None
    return compute_move(instrument, _series(values), vol)


# --- Level formatting ---------------------------------------------------------


class TestFormatLevel:
    def test_large_index_is_not_rendered_in_scientific_notation(self) -> None:
        # The bug this replaced showed the Nikkei as "6.601e+04".
        assert format_level(66008.45, QuoteKind.PRICE) == "66,008.45"

    def test_index_keeps_two_decimals(self) -> None:
        assert format_level(7641.16, QuoteKind.PRICE) == "7,641.16"

    def test_pip_quoted_fx_keeps_four_decimals(self) -> None:
        assert format_level(1.1699, QuoteKind.FX) == "1.1699"

    def test_whole_unit_fx_keeps_two_decimals_not_zero(self) -> None:
        # USD/JPY rounded to "159" throws away the pips a trader watches.
        assert format_level(158.97, QuoteKind.FX) == "158.97"
        assert format_level(1381.26, QuoteKind.FX) == "1,381.26"

    def test_yields_carry_a_percent_sign(self) -> None:
        assert format_level(4.70, QuoteKind.YIELD) == "4.70%"

    def test_column_format_follows_the_block(self) -> None:
        fx_block = [_move(DXY, [1.16, 1.17])]
        assert level_format_for(fx_block) == "%.4f"
        assert level_format_for([_move(SPX, [7600.0, 7641.16])]) == "%.2f"
        assert level_format_for([]) == "%.2f"

    def test_move_exposes_a_formatted_level(self) -> None:
        assert _move(NIKKEI, [66216.79, 66008.45]).display_level == "66,008.45"


# --- Partial sessions ---------------------------------------------------------


class TestPartialSession:
    def test_thin_final_bar_is_flagged(self) -> None:
        # Grains an hour into the open: a real bar on a fraction of the volume.
        volumes = [150_000] * 10 + [30_913]
        assert is_partial_session(_series(volumes)) is True

    def test_normal_volume_is_not_flagged(self) -> None:
        assert is_partial_session(_series([150_000] * 10 + [182_038])) is False

    def test_missing_volume_is_not_evidence_of_a_partial_session(self) -> None:
        # FX, indices and FRED series carry no volume at all.
        assert is_partial_session(None) is False

    def test_zero_volume_bar_is_flagged_as_a_bad_print(self) -> None:
        """A market that normally trades reporting zero volume is a bad bar.

        yfinance occasionally serves one, and it can carry a badly wrong close -
        this is the case that produced a phantom -8.68% move in coffee.
        """
        assert is_partial_session(_series([6000.0] * 10 + [0.0])) is True

    def test_zero_volume_history_does_not_flag_everything(self) -> None:
        # Instruments that never report volume must not all look partial.
        assert is_partial_session(_series([0.0] * 11)) is False

    def test_short_history_is_not_flagged(self) -> None:
        assert is_partial_session(_series([100.0, 5.0])) is False

    def test_partial_move_is_kept_out_of_the_movers_ranking(self) -> None:
        """A thin partial bar must not top the board on volume nobody traded."""
        partial = _move(CORN, [478.75, 501.50], volumes=[150_000] * 10 + [3_000])
        normal = _move(SPX, [7700.0, 7641.16], volumes=[150_000] * 11)

        assert partial.partial_session is True
        ranked = rank_by_magnitude([partial, normal], limit=5)
        assert [m.instrument.symbol for m in ranked] == ["^GSPC"]
        assert all(m.instrument.symbol != "ZC=F" for m in top_gainers([partial, normal]))

    def test_partial_move_is_still_named_in_the_headlines(self) -> None:
        """Excluded from ranking, but not hidden - silence would mislead too."""
        partial = _move(CORN, [478.75, 501.50], volumes=[150_000] * 10 + [3_000])
        facts = headline_facts([partial])
        assert any("mid-session" in fact for fact in facts)


# --- Narrative ----------------------------------------------------------------


class TestNarrative:
    def test_breadth_reports_a_broad_move(self) -> None:
        moves = [_move(SPX, [100.0, 101.0]) for _ in range(4)]
        assert "broadly higher" in describe_breadth(moves)

    def test_breadth_reports_a_mixed_block(self) -> None:
        moves = [_move(SPX, [100.0, 101.0]), _move(SPX, [100.0, 99.0]),
                 _move(SPX, [100.0, 101.0]), _move(SPX, [100.0, 99.0])]
        assert "mixed" in describe_breadth(moves)

    def test_tiny_blocks_get_no_breadth_statistics(self) -> None:
        """'broadly lower (1 of 1 declining)' is a statistic about one number."""
        assert describe_breadth([_move(BRENT, [93.60, 93.55])]) == "little changed"
        assert describe_breadth([_move(BRENT, [90.0, 93.55])]) == "higher"

    def test_empty_block_has_no_directional_read(self) -> None:
        assert describe_breadth([]) == "no directional read"

    def test_vix_does_not_take_the_equity_leader_slot(self) -> None:
        """VIX is implied vol; reporting it as equity strength is wrong."""
        moves = [
            _move(SPX, [7700.0, 7641.16]),  # -0.76%
            _move(RUT, [3033.0, 2992.43]),  # -1.34%
            _move(VIX, [14.89, 16.01]),  # +7.52%
        ]
        line = describe_region(Region.US, SessionState.CLOSED, moves)[0]

        assert "VIX led" not in line
        assert "S&P 500 led" in line
        assert "Volatility spiked" in line

    def test_open_session_uses_the_present_tense(self) -> None:
        moves = [_move(NIKKEI, [66216.79, 66008.45])]
        line = describe_region(Region.APAC, SessionState.OPEN, moves)[0]
        assert "are trading" in line
        assert "Equities is" not in line

    def test_commodity_standout_agrees_with_its_direction(self) -> None:
        """A block called broadly higher must not be illustrated by a faller."""
        moves = [
            _move(GOLD, [4489.0, 4589.20]),
            _move(BRENT, [91.62, 93.53]),
            _move(CORN, [550.0, 501.50]),  # the largest absolute move, downward
        ]
        line = describe_region(Region.GLOBAL, SessionState.OPEN, moves)[-1]

        assert "biggest faller was Corn" in line
        assert "standout" not in line

    def test_divergence_detects_haven_bid(self) -> None:
        moves = [
            _move(SPX, [7700.0, 7641.16]),
            _move(RUT, [3033.0, 2992.43]),
            _move(GOLD, [4489.0, 4589.20]),
        ]
        findings = find_divergences(moves)
        assert any("haven or debasement" in f for f in findings)

    def test_divergence_detects_yields_up_dollar_down(self) -> None:
        moves = [_move(UST10, [4.66, 4.70]), _move(DXY, [98.83, 98.72])]
        findings = find_divergences(moves)
        assert any("fiscal" in f for f in findings)

    def test_divergence_detects_small_cap_underperformance(self) -> None:
        moves = [_move(SPX, [7700.0, 7633.0]), _move(RUT, [3033.0, 2992.43])]
        findings = find_divergences(moves)
        assert any("Small caps lagged" in f for f in findings)

    def test_no_divergence_when_nothing_diverges(self) -> None:
        moves = [_move(SPX, [100.0, 100.05]), _move(RUT, [100.0, 100.04])]
        assert find_divergences(moves) == []


# --- Prompt interpretation ----------------------------------------------------


class TestInterpretPrompt:
    def test_region_is_read_from_an_indirect_mention(self) -> None:
        # "the yen" names neither APAC nor currencies explicitly.
        request = interpret_prompt("brief note on the yen")
        assert Region.APAC in request.regions
        assert AssetClass.CURRENCIES in request.asset_classes

    def test_depth_keywords_are_recognised(self) -> None:
        assert interpret_prompt("quick summary").depth == "brief"
        assert interpret_prompt("deep dive on rates").depth == "deep"
        assert interpret_prompt("what happened today").depth == "standard"

    def test_multiple_regions_are_collected(self) -> None:
        request = interpret_prompt("what moved in UK and Europe")
        assert set(request.regions) == {Region.UK, Region.EUROPE}

    def test_empty_prompt_means_full_scope(self) -> None:
        request = interpret_prompt("")
        assert request.regions == ()
        assert request.asset_classes == ()
        assert request.depth == "standard"

    def test_unmatched_prompt_does_not_narrow_scope(self) -> None:
        """Guessing a narrower scope is worse than covering everything."""
        request = interpret_prompt("tell me something interesting")
        assert request.regions == ()

    def test_substrings_do_not_trigger_a_match(self) -> None:
        # "us" inside "housing" must not select the US region.
        assert Region.US not in interpret_prompt("housing data").regions

    def test_scope_description_is_readable(self) -> None:
        assert "APAC" in interpret_prompt("asia equities").scope_description
        assert "all regions" in interpret_prompt("").scope_description


# --- Pipeline -----------------------------------------------------------------


class FakeMarketProvider(MarketDataProvider):
    name = "fake"

    def get_security_info(self, symbol):
        return SecurityInfo(symbol=symbol, name=symbol)

    def get_daily_bars(self, symbol, start, end):
        bars, price, day = [], 100.0, start
        while day <= end:
            price *= 1.001
            bars.append(
                PriceBar(date=day, close=price, adj_close=price, volume=1_000_000, source="fake")
            )
            day += dt.timedelta(days=1)
        return bars


PIPELINE_UNIVERSE = [SPX, NIKKEI, UST10, GOLD]


@pytest.fixture()
def pipeline(monkeypatch):
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(_env_file=None)
    market_data = MarketDataService(db, FakeMarketProvider(), ApiCache(db), settings)
    monkeypatch.setattr(
        "market_intel.services.recap.instruments_for",
        lambda region=None, asset_class=None: [
            i for i in PIPELINE_UNIVERSE
            if (region is None or i.region is region)
            and (asset_class is None or i.asset_class is asset_class)
        ],
    )
    recap = RecapService(db, market_data, settings)
    return ReportPipeline(recap), recap, db, settings


class TestPipeline:
    def test_report_is_written_without_an_api_key(self, pipeline) -> None:
        """The whole point: no credential still produces prose, not a table."""
        pipe, _, _, _ = pipeline
        assert not pipe.ai_available

        report = pipe.run("full cross-asset wrap")

        assert report.generated_by == "deterministic"
        assert "What the numbers say" in report.content
        assert report.instruments_priced == len(PIPELINE_UNIVERSE)

    def test_deterministic_report_says_it_used_no_model(self, pipeline) -> None:
        pipe, _, _, _ = pipeline
        report = pipe.run("wrap")
        assert "without a language model" in report.content

    def test_prompt_is_echoed_into_the_report(self, pipeline) -> None:
        pipe, _, _, _ = pipeline
        report = pipe.run("deep dive on rates")
        assert "deep dive on rates" in report.content

    def test_scope_narrows_when_the_prompt_names_a_region(self, pipeline) -> None:
        pipe, _, _, _ = pipeline
        report = pipe.run("what happened in Asia")
        assert report.request.regions == (Region.APAC,)
        assert all(m.instrument.region is Region.APAC for m in [])  # scope recorded

    def test_deterministic_report_is_archived(self, pipeline) -> None:
        pipe, _, db, _ = pipeline
        report = pipe.run("wrap")
        assert report.note_id is not None

    def test_a_reused_snapshot_is_not_repriced(self, pipeline) -> None:
        pipe, recap, _, _ = pipeline
        snapshot = recap.build_snapshot()
        calls = []
        object.__setattr__(
            recap, "build_snapshot", lambda *a, **k: calls.append(1) or snapshot
        )

        pipe.run("wrap", snapshot=snapshot)

        assert not calls, "passing a snapshot should skip re-pricing"

    def test_model_failure_falls_back_instead_of_losing_the_report(
        self, pipeline
    ) -> None:
        """A model outage must not cost the user their report."""
        pipe, recap, _, _ = pipeline

        class Boom:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                raise RuntimeError("model exploded")

        recap._client = Boom()
        assert pipe.ai_available

        report = pipe.run("wrap")

        assert report.generated_by == "deterministic"
        assert "AI generation failed" in report.content
        assert "What the numbers say" in report.content

    def test_findings_are_handed_to_the_model_as_established_fact(
        self, pipeline
    ) -> None:
        pipe, recap, _, _ = pipeline
        captured = {}

        class Stub:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                captured.update(kwargs)
                block = type("T", (), {"type": "text", "text": "# Wrap\n\nBody."})()
                return type(
                    "R", (), {"content": [block], "stop_reason": "end_turn",
                              "model": "claude-opus-5"}
                )()

        recap._client = Stub()
        report = pipe.run("deep dive on rates")

        sent = captured["messages"][0]["content"]
        assert "Pre-computed findings" in sent
        assert "deep dive on rates" in sent
        assert report.generated_by == "claude"

    def test_compose_findings_covers_every_region_present(self, pipeline) -> None:
        _, recap, _, _ = pipeline
        findings = compose_findings(recap.build_snapshot())
        for region in (Region.US, Region.APAC, Region.GLOBAL):
            assert region.value in findings


# --- News --------------------------------------------------------------------


class FakeNewsService:
    """Records what it was asked for and returns canned articles."""

    def __init__(self, fail: bool = False) -> None:
        self.refreshed: list[str] = []
        self.fail = fail

    def refresh(self, symbols, limit_per_symbol=20):
        if self.fail:
            raise RuntimeError("news provider down")
        self.refreshed.extend(symbols)
        return len(symbols)

    def get_recent(self, limit=50, symbol=None):
        if self.fail:
            raise RuntimeError("news provider down")
        return [
            {
                "headline": f"{symbol} story",
                "url": f"https://news/{symbol}",
                "source": "Wire",
                "summary": None,
                "published_at": dt.datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
                "symbols": [symbol],
            }
        ]


class TestNews:
    def test_headlines_are_gathered_for_movers(self, pipeline) -> None:
        _, recap, _, _ = pipeline
        service = FakeNewsService()

        articles = gather_headlines(service, recap.build_snapshot())

        assert articles
        assert service.refreshed, "expected a refresh for the mover symbols"

    def test_headlines_are_deduplicated_by_url(self, pipeline) -> None:
        _, recap, _, _ = pipeline

        class Duplicating(FakeNewsService):
            def get_recent(self, limit=50, symbol=None):
                return [
                    {
                        "headline": "Same story",
                        "url": "https://news/same",
                        "source": "Wire",
                        "summary": None,
                        "published_at": None,
                        "symbols": [symbol],
                    }
                ]

        articles = gather_headlines(Duplicating(), recap.build_snapshot())
        assert len(articles) == 1

    def test_news_outage_does_not_fail_the_report(self, pipeline) -> None:
        """A report missing headlines beats a report that never renders."""
        pipe, recap, _, _ = pipeline
        pipe._news = FakeNewsService(fail=True)

        report = pipe.run("wrap")

        assert report.content
        assert report.articles == []

    def test_no_news_service_is_handled(self, pipeline) -> None:
        _, recap, _, _ = pipeline
        assert gather_headlines(None, recap.build_snapshot()) == []

    def test_report_carries_a_top_news_section(self, pipeline) -> None:
        pipe, _, _, _ = pipeline
        pipe._news = FakeNewsService()

        report = pipe.run("wrap")

        assert "## Top news" in report.content
        assert report.articles

    def test_news_can_be_switched_off(self, pipeline) -> None:
        pipe, _, _, _ = pipeline
        pipe._news = FakeNewsService()

        report = pipe.run("wrap", include_news=False)

        assert report.articles == []
        assert "## Top news" not in report.content

    def test_headlines_reach_the_model_prompt(self, pipeline) -> None:
        pipe, recap, _, _ = pipeline
        pipe._news = FakeNewsService()
        captured = {}

        class Stub:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                captured.update(kwargs)
                block = type("T", (), {"type": "text", "text": "# Wrap"})()
                return type(
                    "R", (), {"content": [block], "stop_reason": "end_turn",
                              "model": "claude-opus-5"}
                )()

        recap._client = Stub()
        pipe.run("wrap")

        assert "Headlines already collected" in captured["messages"][0]["content"]

    def test_format_headlines_is_empty_for_no_articles(self) -> None:
        assert format_headlines([]) == ""

    def test_format_headlines_renders_links_and_attribution(self) -> None:
        rendered = format_headlines(
            [
                {
                    "headline": "Gold rallies",
                    "url": "https://news/gold",
                    "source": "Reuters",
                    "published_at": dt.datetime(2026, 8, 20, 9, 30, tzinfo=UTC),
                    "symbols": ["GC=F"],
                }
            ]
        )
        assert "[Gold rallies](https://news/gold)" in rendered
        assert "Reuters" in rendered
        assert "GC=F" in rendered
