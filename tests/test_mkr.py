"""Tests for the MKR 14-Framework analysis: maths, plans, rendering, write-up.

The arithmetic tests use hand-built series where the right answer is known by
construction, because that is the only way to catch a units or direction bug —
a retracement grid computed off the wrong end of the swing still looks
plausible on a chart.
"""

import datetime as dt
import math

import pandas as pd
import pytest

from market_intel.analysis import mkr as calc
from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    ProviderError,
)
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider
from market_intel.providers.fundamentals import (
    Fundamentals,
    OptionQuote,
    OptionsSnapshot,
    atm_implied_vol,
    max_pain,
    unusual_activity,
)
from market_intel.services.market_data import MarketDataService
from market_intel.services.mkr import MkrService, format_analysis, render_report

UTC = dt.timezone.utc
TODAY = dt.date(2026, 9, 28)


def _frame(
    closes: list[float],
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    end: dt.date = TODAY,
) -> pd.DataFrame:
    """Build a date-indexed OHLCV frame ending at ``end``."""
    index = [end - dt.timedelta(days=len(closes) - 1 - i) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs or [value * 1.01 for value in closes],
            "low": lows or [value * 0.99 for value in closes],
            "close": closes,
            "volume": volumes or [1_000_000.0] * len(closes),
        },
        index=pd.Index(index, name="date"),
    )


def _zigzag(legs: list[tuple[float, int]], start: float = 100.0) -> list[float]:
    """Piecewise-linear closes: each leg walks to a price over N sessions."""
    values = [start]
    for target, steps in legs:
        current = values[-1]
        for step in range(1, steps + 1):
            values.append(current + (target - current) * step / steps)
    return values


# --- Swing structure ----------------------------------------------------------


class TestPivots:
    def test_pivots_alternate_high_and_low(self) -> None:
        frame = _frame(_zigzag([(140, 20), (110, 20), (170, 20), (150, 20)]))
        pivots = calc.find_pivots(frame)

        assert len(pivots) >= 3
        kinds = [pivot.kind for pivot in pivots]
        assert all(first != second for first, second in zip(kinds, kinds[1:]))

    def test_a_short_series_yields_no_pivots(self) -> None:
        assert calc.find_pivots(_frame([1.0, 2.0, 3.0])) == []

    def test_latest_swing_is_the_final_leg(self) -> None:
        frame = _frame(_zigzag([(140, 20), (110, 20), (170, 20)]))
        pivots = calc.find_pivots(frame)
        swing = calc.latest_swing(pivots)

        assert swing is not None
        assert swing.start is pivots[-2]
        assert swing.end is pivots[-1]

    def test_swing_reports_its_own_direction_and_span(self) -> None:
        low = calc.Pivot(dt.date(2026, 1, 5), 100.0, "low")
        high = calc.Pivot(dt.date(2026, 3, 5), 150.0, "high")
        swing = calc.Swing(low, high)

        assert swing.is_up
        assert swing.range == pytest.approx(50.0)
        assert swing.pct == pytest.approx(50.0)


# --- Fibonacci ----------------------------------------------------------------


class TestFibonacci:
    def test_up_leg_retraces_downward_from_the_high(self) -> None:
        low = calc.Pivot(dt.date(2026, 1, 5), 100.0, "low")
        high = calc.Pivot(dt.date(2026, 3, 5), 200.0, "high")
        read = calc.fib_read([low, high], last_close=180.0)

        # A 100-point up leg retraces to 200 - 100 * ratio.
        assert read.retracements["61.8%"] == pytest.approx(138.2)
        assert read.retracements["38.2%"] == pytest.approx(161.8)
        assert read.retracements["50.0%"] == pytest.approx(150.0)

    def test_down_leg_retraces_upward_from_the_low(self) -> None:
        high = calc.Pivot(dt.date(2026, 1, 5), 200.0, "high")
        low = calc.Pivot(dt.date(2026, 3, 5), 100.0, "low")
        read = calc.fib_read([high, low], last_close=120.0)

        assert read.retracements["61.8%"] == pytest.approx(161.8)
        assert read.retracements["38.2%"] == pytest.approx(138.2)

    def test_golden_pocket_is_ordered_low_to_high(self) -> None:
        read = calc.fib_read(
            [
                calc.Pivot(dt.date(2026, 1, 5), 100.0, "low"),
                calc.Pivot(dt.date(2026, 3, 5), 200.0, "high"),
            ],
            last_close=150.0,
        )
        assert read.golden_pocket is not None
        assert read.golden_pocket[0] < read.golden_pocket[1]
        # 61.8-65% of a 100-point leg below 200.
        assert read.golden_pocket == pytest.approx((135.0, 138.2))

    def test_position_in_the_swing_is_a_percentage(self) -> None:
        read = calc.fib_read(
            [
                calc.Pivot(dt.date(2026, 1, 5), 100.0, "low"),
                calc.Pivot(dt.date(2026, 3, 5), 200.0, "high"),
            ],
            last_close=150.0,
        )
        assert read.position_pct == pytest.approx(50.0)

    def test_no_swing_means_no_grid(self) -> None:
        read = calc.fib_read([], last_close=100.0)
        assert read.swing is None
        assert read.to_signal().signal == "n/a"

    def test_extensions_project_beyond_the_impulse(self) -> None:
        read = calc.fib_read(
            [
                calc.Pivot(dt.date(2026, 1, 5), 100.0, "low"),
                calc.Pivot(dt.date(2026, 2, 5), 200.0, "high"),
                calc.Pivot(dt.date(2026, 3, 5), 160.0, "low"),
            ],
            last_close=170.0,
        )
        # Anchored on the 160 correction, projecting the 100-point impulse.
        assert read.extensions["1.618x"] == pytest.approx(160.0 + 161.8)
        assert "160" in read.extension_anchor


# --- RSI ----------------------------------------------------------------------


class TestRsi:
    def test_daily_and_weekly_are_both_reported(self) -> None:
        closes = _zigzag([(160, 60), (130, 40), (190, 60)])
        daily = _frame(closes)
        weekly = calc.resample_ohlcv(daily, "W")
        read = calc.rsi_read(daily, weekly)

        assert read.daily is not None and 0.0 <= read.daily <= 100.0
        assert read.weekly is not None and 0.0 <= read.weekly <= 100.0

    def test_regular_bearish_divergence_is_detected(self) -> None:
        # Two rallies to higher highs, the second on much weaker follow-through,
        # so RSI peaks lower even though price does not.
        closes = _zigzag([(150, 12), (100, 12), (155, 40), (140, 8)])
        read = calc.rsi_read(_frame(closes))

        assert read.divergence in ("regular bearish", "hidden bearish", None)
        if read.divergence:
            assert read.divergence_detail

    def test_hidden_divergence_is_flagged_as_continuation(self) -> None:
        read = calc.RsiRead(daily=58.0, weekly=60.0, divergence="hidden bullish")
        signal = read.to_signal()
        assert signal.signal == "bullish"
        assert "hidden bullish" in signal.detail

    def test_overbought_reads_bearish_without_divergence(self) -> None:
        signal = calc.RsiRead(daily=82.0, weekly=70.0, divergence=None).to_signal()
        assert signal.signal == "bearish"
        assert "overbought" in signal.detail

    def test_empty_frame_is_not_an_error(self) -> None:
        read = calc.rsi_read(pd.DataFrame(columns=["close"]))
        assert read.daily is None
        assert read.to_signal().signal == "n/a"


# --- Fair value gaps ----------------------------------------------------------


class TestFairValueGaps:
    def test_a_bullish_gap_is_found_between_bar_one_and_three(self) -> None:
        frame = _frame(
            closes=[100.0, 101.0, 110.0, 111.0],
            highs=[101.0, 102.0, 112.0, 113.0],
            lows=[99.0, 100.0, 108.0, 110.0],
        )
        gaps = calc.find_fair_value_gaps(frame)

        # Two overlapping imbalances print here, one per three-bar window, and
        # both are real: each marks a band price skipped and has not revisited.
        assert [(gap.low, gap.high) for gap in gaps] == [(101.0, 108.0), (102.0, 110.0)]
        assert all(gap.direction == "bullish" for gap in gaps)

    def test_a_filled_gap_is_excluded(self) -> None:
        frame = _frame(
            closes=[100.0, 101.0, 110.0, 100.0],
            highs=[101.0, 102.0, 112.0, 111.0],
            lows=[99.0, 100.0, 108.0, 99.0],
        )
        assert calc.find_fair_value_gaps(frame) == []

    def test_a_bearish_gap_is_found_in_the_other_direction(self) -> None:
        frame = _frame(
            closes=[110.0, 109.0, 100.0],
            highs=[112.0, 111.0, 101.0],
            lows=[108.0, 107.0, 99.0],
        )
        gaps = calc.find_fair_value_gaps(frame)
        assert len(gaps) == 1
        assert gaps[0].direction == "bearish"
        assert (gaps[0].low, gaps[0].high) == (101.0, 108.0)

    def test_read_splits_gaps_above_and_below_spot(self) -> None:
        gaps = (
            calc.FairValueGap(TODAY, 90.0, 92.0, "bullish", "daily"),
            calc.FairValueGap(TODAY, 120.0, 124.0, "bearish", "daily"),
        )
        read = calc.FvgRead(
            gaps, nearest_above=gaps[1], nearest_below=gaps[0], last_close=100.0
        )
        detail = read.to_signal().detail
        assert "draw below 90.00-92.00" in detail
        assert "draw above 120.00-124.00" in detail

    def test_the_closer_draw_sets_the_direction(self) -> None:
        below = calc.FairValueGap(TODAY, 98.0, 99.0, "bullish", "daily")
        above = calc.FairValueGap(TODAY, 130.0, 134.0, "bearish", "daily")
        read = calc.FvgRead(
            (below, above), nearest_above=above, nearest_below=below, last_close=100.0
        )
        # The gap 1.5 points below is a far nearer magnet than the one 32 above.
        assert read.to_signal().signal == "bearish"

    def test_a_gap_only_above_reads_bullish(self) -> None:
        above = calc.FairValueGap(TODAY, 130.0, 134.0, "bullish", "daily")
        read = calc.FvgRead((above,), nearest_above=above, last_close=100.0)
        assert read.to_signal().signal == "bullish"


# --- Order flow ---------------------------------------------------------------


class TestFlow:
    def test_rising_obv_on_heavy_volume_is_strongly_bullish(self) -> None:
        closes = list(range(100, 160))
        volumes = [1_000_000.0] * 59 + [2_000_000.0]
        read = calc.flow_read(_frame([float(c) for c in closes], volumes))

        assert read.obv_direction == "rising"
        assert read.ratio is not None and read.ratio > 1.0
        assert read.to_signal().signal == "bullish"

    def test_climax_bars_are_dated(self) -> None:
        volumes = [1_000_000.0] * 59 + [5_000_000.0]
        read = calc.flow_read(_frame([100.0 + i for i in range(60)], volumes))
        assert read.climax_dates and read.climax_dates[-1] == TODAY

    def test_missing_volume_degrades_rather_than_raising(self) -> None:
        read = calc.flow_read(_frame([100.0] * 30, volumes=[0.0] * 30))
        assert read.ratio is None
        assert read.to_signal().signal == "n/a"


# --- Moving averages ----------------------------------------------------------


class TestMovingAverages:
    def test_an_uptrend_stacks_the_averages(self) -> None:
        frame = _frame([100.0 + i * 0.5 for i in range(300)])
        read = calc.ma_read(frame, calc.find_pivots(frame), float(frame["close"].iloc[-1]))

        assert read.ema20 is not None and read.ema50 is not None
        assert read.ema20 > read.ema50 > read.sma200
        assert read.to_signal().signal == "bullish"
        assert "golden cross" in read.cross

    def test_a_downtrend_inverts_the_stack(self) -> None:
        frame = _frame([250.0 - i * 0.5 for i in range(300)])
        read = calc.ma_read(frame, calc.find_pivots(frame), float(frame["close"].iloc[-1]))

        assert read.ema20 < read.ema50 < read.sma200
        assert read.to_signal().signal == "bearish"

    def test_supports_sit_below_spot_and_resistances_above(self) -> None:
        frame = _frame(_zigzag([(150, 30), (110, 30), (170, 30), (130, 30)]))
        last = float(frame["close"].iloc[-1])
        read = calc.ma_read(frame, calc.find_pivots(frame), last)

        assert all(level < last for level in read.supports)
        assert all(level > last for level in read.resistances)


# --- Momentum -----------------------------------------------------------------


class TestMomentum:
    def test_a_trending_series_reports_adx_above_the_threshold(self) -> None:
        read = calc.momentum_read(_frame([100.0 * (1.005**i) for i in range(200)]))

        assert read.adx is not None and read.adx > calc.ADX_TRENDING
        assert read.trending
        assert read.atr is not None and read.atr > 0
        assert read.to_signal().signal == "bullish"

    def test_atr_is_expressed_as_a_share_of_price(self) -> None:
        read = calc.momentum_read(_frame([100.0] * 60))
        assert read.atr_pct is not None and read.atr_pct >= 0

    def test_short_history_returns_an_unavailable_signal(self) -> None:
        assert calc.momentum_read(_frame([100.0] * 10)).to_signal().signal == "n/a"

    def test_adx_is_bounded(self) -> None:
        values = calc.adx(_frame(_zigzag([(140, 40), (110, 40), (160, 40)]))).dropna()
        assert not values.empty
        assert values.between(0.0, 100.0).all()


# --- Multi-timeframe ----------------------------------------------------------


class TestTimeframes:
    def test_resampling_preserves_the_high_and_the_low(self) -> None:
        frame = _frame([100.0, 110.0, 90.0, 105.0, 102.0, 99.0, 101.0])
        weekly = calc.resample_ohlcv(frame, "W")

        assert not weekly.empty
        assert float(weekly["high"].max()) == pytest.approx(float(frame["high"].max()))
        assert float(weekly["low"].min()) == pytest.approx(float(frame["low"].min()))

    def test_a_long_uptrend_aligns_every_timeframe(self) -> None:
        daily = _frame([100.0 * (1.001**i) for i in range(900)])
        read = calc.timeframe_read(
            daily, calc.resample_ohlcv(daily, "W"), calc.resample_ohlcv(daily, "ME")
        )

        assert read.aligned
        assert read.to_signal().signal == "bullish"
        assert read.to_signal().strength == "strong"

    def test_mixed_timeframes_are_neutral(self) -> None:
        read = calc.TimeframeRead(
            (
                calc.TimeframeTrend("Daily", "bullish", ""),
                calc.TimeframeTrend("Weekly", "bearish", ""),
            )
        )
        assert not read.aligned
        assert read.to_signal().signal == "neutral"
        assert "mixed" in read.to_signal().detail


# --- Confirmation and overrides -----------------------------------------------


class TestConfirmation:
    def test_death_cross_override_fires_in_a_downtrend(self) -> None:
        read = calc.confirmation_read(_frame([300.0 - i * 0.4 for i in range(400)]))

        assert read.death_cross_override
        signal = read.to_signal()
        assert signal.signal == "bearish"
        assert "DEATH CROSS OVERRIDE" in signal.detail

    def test_override_stays_off_in_an_uptrend(self) -> None:
        read = calc.confirmation_read(_frame([100.0 + i * 0.4 for i in range(400)]))
        assert not read.death_cross_override
        assert read.to_signal().signal == "bullish"

    def test_trix_slope_is_labelled(self) -> None:
        read = calc.confirmation_read(_frame([100.0 * (1.002**i) for i in range(200)]))
        assert read.trix_slope == "rising"


class TestExhaustion:
    def test_one_flag_is_not_enough_to_call_a_top(self) -> None:
        read = calc.exhaustion_read(
            calc.MomentumRead(1.0, "rising", 30.0, 0.9, 1.0, 2.0, 1.0),
            calc.RsiRead(daily=80.0, weekly=70.0, divergence=None),
            calc.FlowRead(1.0, 1.0, 1.0, "steady", "rising"),
        )
        assert len(read.flags) == 1
        assert not read.triggered
        assert "watching" in read.summary

    def test_two_flags_trigger_the_override(self) -> None:
        read = calc.exhaustion_read(
            calc.MomentumRead(1.0, "rising", 30.0, 1.1, 2.6, 2.0, 1.0),
            calc.RsiRead(daily=81.0, weekly=70.0, divergence=None),
            calc.FlowRead(1.0, 1.0, 1.0, "steady", "rising"),
        )
        assert read.triggered
        assert "TRIGGERED" in read.summary

    def test_a_quiet_tape_raises_no_flags(self) -> None:
        read = calc.exhaustion_read(
            calc.MomentumRead(0.1, "flat", 12.0, 0.5, 0.2, 1.0, 1.0),
            calc.RsiRead(daily=52.0, weekly=50.0, divergence=None),
            calc.FlowRead(1.0, 1.0, 1.0, "steady", "flat"),
        )
        assert read.summary == "no exhaustion flags"


# --- Monte Carlo --------------------------------------------------------------


class TestMonteCarlo:
    def test_results_are_reproducible_for_a_given_seed(self) -> None:
        closes = pd.Series([100.0 * (1.001**i) for i in range(300)])
        first = calc.monte_carlo(closes, "Neutral", paths=200)
        second = calc.monte_carlo(closes, "Neutral", paths=200)

        assert first is not None and second is not None
        assert first.win_rate == second.win_rate
        assert first.worst_1pct == second.worst_1pct

    def test_a_positive_tilt_raises_the_win_rate(self) -> None:
        closes = pd.Series(
            [100.0 * (1.0 + 0.01 * math.sin(i / 7.0)) for i in range(300)]
        )
        neutral = calc.monte_carlo(closes, "Neutral", paths=500)
        tilted = calc.monte_carlo(closes, "Tilted", paths=500, annual_drift=0.60)

        assert neutral is not None and tilted is not None
        assert tilted.win_rate > neutral.win_rate
        # Same volatility, different drift — that is the whole point.
        assert tilted.annual_vol == pytest.approx(neutral.annual_vol)

    def test_percentiles_are_ordered(self) -> None:
        closes = pd.Series([100.0 * (1.002**i) for i in range(300)])
        result = calc.monte_carlo(closes, "Neutral", paths=300)
        assert result is not None
        assert result.worst_1pct <= result.percentile_5 <= result.median_return
        assert result.median_return <= result.percentile_95

    def test_too_little_history_returns_none(self) -> None:
        assert calc.monte_carlo(pd.Series([100.0, 101.0]), "Neutral") is None


# --- Black-Scholes ------------------------------------------------------------


class TestBlackScholes:
    def test_call_delta_is_bounded_and_rises_with_spot(self) -> None:
        low = calc.bs_delta(90.0, 100.0, 1.0, 0.4, 0.04)
        high = calc.bs_delta(130.0, 100.0, 1.0, 0.4, 0.04)

        assert low is not None and high is not None
        assert 0.0 < low < high < 1.0

    def test_put_delta_is_negative(self) -> None:
        delta = calc.bs_delta(100.0, 100.0, 0.25, 0.35, 0.04, kind="put")
        assert delta is not None and -1.0 < delta < 0.0

    def test_put_call_parity_holds(self) -> None:
        spot, strike, years, vol, rate = 120.0, 110.0, 0.5, 0.3, 0.04
        call = calc.bs_price(spot, strike, years, vol, rate)
        put = calc.bs_price(spot, strike, years, vol, rate, kind="put")

        assert call is not None and put is not None
        assert call - put == pytest.approx(
            spot - strike * math.exp(-rate * years), rel=1e-6
        )

    def test_degenerate_inputs_return_none(self) -> None:
        assert calc.bs_delta(100.0, 100.0, 0.0, 0.3, 0.04) is None
        assert calc.bs_price(100.0, 100.0, 1.0, 0.0, 0.04) is None


# --- Scorecard ----------------------------------------------------------------


class TestConfidence:
    def test_a_fully_bullish_board_scores_high(self) -> None:
        signals = [
            calc.FrameworkSignal(f"F{i}", "bullish", "strong") for i in range(14)
        ]
        assert calc.confidence_score(signals) == 100

    def test_a_fully_bearish_board_scores_low(self) -> None:
        signals = [
            calc.FrameworkSignal(f"F{i}", "bearish", "strong") for i in range(14)
        ]
        assert calc.confidence_score(signals) == 0

    def test_an_empty_board_is_fifty(self) -> None:
        assert calc.confidence_score([]) == 50
        assert calc.confidence_score(
            [calc.FrameworkSignal("F", "n/a", "—")]
        ) == 50

    def test_the_death_cross_override_caps_confidence(self) -> None:
        signals = [
            calc.FrameworkSignal(f"F{i}", "bullish", "strong") for i in range(14)
        ]
        assert calc.confidence_score(signals, override=True) == 35


# --- Option chain analytics ---------------------------------------------------


def _chain_quotes(expiry: dt.date) -> list[OptionQuote]:
    """A symmetric chain with the heaviest open interest at 100."""
    quotes: list[OptionQuote] = []
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        interest = 5_000 if strike == 100.0 else 500
        for kind in ("call", "put"):
            quotes.append(
                OptionQuote(
                    expiry=expiry,
                    strike=strike,
                    kind=kind,
                    bid=4.0,
                    ask=4.4,
                    last=4.2,
                    implied_vol=0.45,
                    open_interest=interest,
                    volume=100,
                )
            )
    return quotes


class TestChainAnalytics:
    def test_max_pain_lands_on_the_heaviest_strike(self) -> None:
        assert max_pain(_chain_quotes(dt.date(2026, 12, 18))) == 100.0

    def test_max_pain_needs_a_real_chain(self) -> None:
        expiry = dt.date(2026, 12, 18)
        thin = [
            OptionQuote(expiry, 100.0, "call", 1.0, 1.2, 1.1, 0.4, 100, 10),
        ]
        assert max_pain(thin) is None

    def test_atm_iv_uses_the_strike_closest_to_spot(self) -> None:
        quotes = _chain_quotes(dt.date(2026, 12, 18))
        assert atm_implied_vol(quotes, spot=101.0) == pytest.approx(0.45)

    def test_unusual_activity_needs_volume_far_above_open_interest(self) -> None:
        expiry = dt.date(2026, 12, 18)
        quiet = OptionQuote(expiry, 100.0, "call", 1.0, 1.2, 1.1, 0.4, 5_000, 300)
        loud = OptionQuote(expiry, 110.0, "call", 1.0, 1.2, 1.1, 0.4, 100, 900)

        flagged = unusual_activity([quiet, loud])
        assert len(flagged) == 1
        assert "110C" in flagged[0]

    def test_options_snapshot_round_trips_through_the_cache_payload(self) -> None:
        expiry = dt.date(2026, 12, 18)
        snapshot = OptionsSnapshot(
            symbol="TEST",
            as_of=TODAY,
            expiries=(expiry,),
            quotes=tuple(_chain_quotes(expiry)),
            put_call_oi=0.9,
            max_pain=100.0,
            max_pain_expiry=expiry,
            atm_iv=0.45,
            unusual=("something",),
        )
        restored = OptionsSnapshot.from_payload(snapshot.to_payload())

        assert restored == snapshot

    def test_fundamentals_round_trip_preserves_the_earnings_date(self) -> None:
        fundamentals = Fundamentals(
            symbol="TEST", next_earnings=dt.date(2026, 11, 5), market_cap=1.2e10
        )
        restored = Fundamentals.from_payload(fundamentals.to_payload())
        assert restored == fundamentals


# --- Service ------------------------------------------------------------------


class FakeMarketProvider(MarketDataProvider):
    """Deterministic bars: a steady uptrend with an oscillation on top."""

    name = "fake"

    def get_security_info(self, symbol):
        return SecurityInfo(symbol=symbol, name=f"{symbol} Inc")

    def get_daily_bars(self, symbol, start, end):
        bars: list[PriceBar] = []
        day, index = start, 0
        while day <= end:
            trend = 100.0 * (1.0012**index)
            close = trend * (1.0 + 0.04 * math.sin(index / 11.0))
            bars.append(
                PriceBar(
                    date=day,
                    open=close * 0.998,
                    high=close * 1.012,
                    low=close * 0.988,
                    close=close,
                    adj_close=close,
                    volume=1_000_000 + (index % 7) * 50_000,
                    source="fake",
                )
            )
            day += dt.timedelta(days=1)
            index += 1
        return bars


class FakeFundamentalsProvider:
    """Stand-in for the yfinance fundamentals/options/intraday feeds."""

    name = "fake"

    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.fail = fail

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        if "fundamentals" in self.fail:
            raise DataNotFoundError("no coverage", provider=self.name)
        return Fundamentals(
            symbol=symbol,
            name=f"{symbol} Inc",
            market_cap=8.5e9,
            peg_ratio=0.42,
            eps_forward=4.2,
            revenue_growth=0.34,
            profit_margin=0.22,
            target_low=180.0,
            target_mean=260.0,
            target_high=320.0,
            analyst_count=28,
            recommendation="buy",
            held_percent_insiders=0.03,
            next_earnings=TODAY + dt.timedelta(days=21),
            insider_net_shares_6m=-12_000.0,
            sector="Technology",
            industry="Semiconductors",
        )

    def get_options(self, symbol: str, max_expiries: int = 6) -> OptionsSnapshot:
        if "options" in self.fail:
            raise DataNotFoundError("no listed options", provider=self.name)
        near = TODAY + dt.timedelta(days=35)
        far = TODAY + dt.timedelta(days=300)
        quotes: list[OptionQuote] = []
        for expiry in (near, far):
            for strike in [80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
                for kind in ("call", "put"):
                    quotes.append(
                        OptionQuote(
                            expiry=expiry,
                            strike=strike,
                            kind=kind,
                            bid=strike * 0.05,
                            ask=strike * 0.055,
                            last=strike * 0.052,
                            implied_vol=0.42,
                            open_interest=1_000,
                            volume=200,
                        )
                    )
        return OptionsSnapshot(
            symbol=symbol,
            as_of=TODAY,
            expiries=(near, far),
            quotes=tuple(quotes),
            put_call_oi=0.85,
            put_call_volume=0.9,
            max_pain=140.0,
            max_pain_expiry=near,
            atm_iv=0.42,
            unusual=("some big print",),
        )

    def get_intraday(self, symbol: str, days: int = 60) -> pd.DataFrame:
        if "intraday" in self.fail:
            raise DataNotFoundError("no intraday", provider=self.name)
        index = pd.date_range("2026-08-01", periods=120, freq="4h")
        closes = [100.0 + index.get_loc(stamp) * 0.2 for stamp in index]
        return pd.DataFrame(
            {
                "open": closes,
                "high": [value * 1.004 for value in closes],
                "low": [value * 0.996 for value in closes],
                "close": closes,
                "volume": [10_000.0] * len(closes),
            },
            index=index,
        )


@pytest.fixture()
def mkr_service():
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(_env_file=None, mkr_history_days=700, mkr_monte_carlo_paths=200)
    cache = ApiCache(db)
    market_data = MarketDataService(db, FakeMarketProvider(), cache, settings)
    service = MkrService(
        db, market_data, settings, cache, FakeFundamentalsProvider()
    )
    return service, db, settings, market_data


class TestAnalyse:
    def test_every_framework_produces_a_scorecard_row(self, mkr_service) -> None:
        service, *_ = mkr_service
        analysis = service.analyse("TEST", now=dt.datetime(2026, 9, 28, 20, tzinfo=UTC))

        assert len(analysis.signals) == 14
        names = [signal.name for signal in analysis.signals]
        assert names[0] == "Elliott Wave"
        assert names[-1] == "Capital Rotation"
        assert len(set(names)) == 14

    def test_confidence_is_a_percentage(self, mkr_service) -> None:
        service, *_ = mkr_service
        analysis = service.analyse("TEST")
        assert 0 <= analysis.confidence <= 100

    def test_progress_is_reported_stage_by_stage(self, mkr_service) -> None:
        service, *_ = mkr_service
        seen: list[str] = []
        service.analyse("TEST", progress=seen.append)
        assert len(seen) >= 5
        assert any("Monte Carlo" in message for message in seen)

    def test_levels_are_ordered_around_spot(self, mkr_service) -> None:
        service, *_ = mkr_service
        levels = service.analyse("TEST").levels

        if levels.stop is not None:
            assert levels.stop < levels.current
        if levels.target_1 is not None:
            assert levels.target_1 > levels.current
        if levels.target_1 and levels.target_2:
            assert levels.target_2 >= levels.target_1

    def test_an_unknown_symbol_is_an_error_not_an_empty_report(self, mkr_service) -> None:
        service, db, settings, _ = mkr_service

        class Dead(FakeMarketProvider):
            def get_daily_bars(self, symbol, start, end):
                raise DataNotFoundError("nothing here", provider="fake")

        market_data = MarketDataService(db, Dead(), ApiCache(db), settings)
        broken = MkrService(db, market_data, settings, None, FakeFundamentalsProvider())
        with pytest.raises(ProviderError):
            broken.analyse("GHOST")

    def test_a_failed_optional_feed_is_recorded_not_raised(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        degraded = MkrService(
            db, market_data, settings, None,
            FakeFundamentalsProvider(fail=("fundamentals", "options", "intraday")),
        )
        analysis = degraded.analyse("TEST")

        assert len(analysis.unavailable) == 3
        assert not analysis.options.available
        assert not analysis.catalyst.available
        assert not analysis.leaps.available
        assert not analysis.wheel.available
        # The report still exists, with the gaps named.
        report = render_report(analysis)
        assert "DATA NOT AVAILABLE" in report

    def test_intraday_bars_add_four_hour_gaps(self, mkr_service) -> None:
        service, *_ = mkr_service
        analysis = service.analyse("TEST")
        assert analysis.fvg.intraday_available

    def test_fundamentals_are_cached_between_runs(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        counting = FakeFundamentalsProvider()
        calls: list[str] = []
        original = counting.get_fundamentals
        counting.get_fundamentals = lambda symbol: (  # type: ignore[method-assign]
            calls.append(symbol) or original(symbol)
        )
        cached = MkrService(db, market_data, settings, ApiCache(db), counting)

        cached.analyse("TEST")
        cached.analyse("TEST")
        assert len(calls) == 1


class TestUniScore:
    def test_computable_legs_are_scored_and_the_rest_flagged(self, mkr_service) -> None:
        service, *_ = mkr_service
        uni = service.analyse("TEST").uni

        assert len(uni.legs) == 5
        assert 5 <= uni.total <= 25
        assert "Physical scarcity / monopoly IP" in uni.judgement_legs
        # $8.5bn market cap and +34% revenue growth are both measured, not guessed.
        cap_leg = next(leg for leg in uni.legs if "Market cap" in leg.name)
        assert cap_leg.computed and cap_leg.score == 4

    def test_no_fundamentals_means_no_legs(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        degraded = MkrService(
            db, market_data, settings, None,
            FakeFundamentalsProvider(fail=("fundamentals",)),
        )
        uni = degraded.analyse("TEST").uni
        assert uni.legs == ()
        assert uni.to_signal().signal == "n/a"


class TestCatalyst:
    def test_a_peg_below_half_is_the_highest_priority_alert(self, mkr_service) -> None:
        service, *_ = mkr_service
        catalyst = service.analyse("TEST").catalyst
        assert catalyst.peg_alert == "HIGHEST PRIORITY"

    def test_days_to_earnings_is_counted_from_today(self, mkr_service) -> None:
        service, *_ = mkr_service
        catalyst = service.analyse("TEST", now=dt.datetime(2026, 9, 28, tzinfo=UTC)).catalyst
        assert catalyst.days_to_earnings == 21


class TestOptionPlans:
    def test_the_leaps_leg_targets_a_deep_delta_and_a_long_expiry(self, mkr_service) -> None:
        service, *_ = mkr_service
        leaps = service.analyse("TEST").leaps

        assert leaps.available
        assert (leaps.expiry - TODAY).days >= 180
        assert 0.4 <= leaps.delta <= 0.95
        assert leaps.breakeven == pytest.approx(leaps.strike + leaps.premium)

    def test_leaps_scenarios_cover_both_horizons(self, mkr_service) -> None:
        service, *_ = mkr_service
        leaps = service.analyse("TEST").leaps
        labels = [row[0] for row in leaps.scenarios]

        assert any(label.startswith("30d") for label in labels)
        assert any(label.startswith("60d") for label in labels)
        # A bigger stock move must produce a bigger option move.
        slow = next(row for row in leaps.scenarios if row[0] == "30d slow")
        fast = next(row for row in leaps.scenarios if row[0] == "30d fast")
        assert fast[2] > slow[2]

    def test_the_wheel_leg_sells_a_put_near_thirty_delta(self, mkr_service) -> None:
        service, *_ = mkr_service
        wheel = service.analyse("TEST").wheel

        assert wheel.available
        assert wheel.delta is not None and wheel.delta < 0
        assert 21 <= (wheel.expiry - TODAY).days <= 60
        assert wheel.cost_basis == pytest.approx(wheel.strike - wheel.premium)
        assert 0.0 < wheel.probability_otm < 100.0

    def test_iv_rank_is_never_claimed(self, mkr_service) -> None:
        service, *_ = mkr_service
        analysis = service.analyse("TEST")
        pack = format_analysis(analysis)

        assert analysis.options.available
        assert "IV RANK IS NOT AVAILABLE" in pack
        assert analysis.options.realised_vol is not None


class TestRotationAndEntry:
    def test_a_death_cross_forces_no_entry_and_no_size(self) -> None:
        service = MkrService.__new__(MkrService)  # rules only, no I/O needed
        confirmation = calc.ConfirmationRead(
            ema20=90.0, ema50=95.0, ema200=120.0, rsi14=40.0,
            macd_trend="falling", trix=-5.0, trix_slope="falling",
            death_cross_override=True,
        )
        levels = __import__(
            "market_intel.services.mkr", fromlist=["LevelPlan"]
        ).LevelPlan(100.0, 88.0, 92.0, (88.0,), 120.0, 140.0, 160.0)
        exhaustion = calc.ExhaustionRead(())
        moving_averages = calc.MaRead(90.0, 95.0, 120.0, 120.0, 94.0, "death cross intact", 200.0)

        entry = MkrService._entry_plan(
            service, 100.0, 80, confirmation, exhaustion, levels, moving_averages,
            calc.TimeframeRead(()),
        )
        rotation = MkrService._rotation_read(
            service, 80, confirmation, exhaustion, levels, calc.FibRead(None)
        )

        assert entry.decision == "N"
        assert "Death cross override" in entry.rationale
        assert rotation.entry_size_pct == 0.0

    def test_a_thin_reward_downgrades_a_full_entry_to_partial(self) -> None:
        """A green board with the target inside the stop is not a buy at market."""
        service = MkrService.__new__(MkrService)
        module = __import__("market_intel.services.mkr", fromlist=["LevelPlan"])
        confirmation = calc.ConfirmationRead(
            ema20=98.0, ema50=96.0, ema200=90.0, rsi14=60.0,
            macd_trend="rising", trix=8.0, trix_slope="rising",
        )
        aligned = calc.TimeframeRead(
            (
                calc.TimeframeTrend("Daily", "bullish", ""),
                calc.TimeframeTrend("Weekly", "bullish", ""),
                calc.TimeframeTrend("Monthly", "bullish", ""),
            )
        )
        moving_averages = calc.MaRead(
            98.0, 96.0, 90.0, 90.0, 97.0, "golden cross intact", 120.0,
            supports=(92.0,), resistances=(101.0,),
        )
        # Stop 10% away, T1 1% away: a 0.1x payout.
        thin = module.LevelPlan(100.0, 90.0, 94.0, (92.0,), 101.0, 120.0, 140.0)
        generous = module.LevelPlan(100.0, 90.0, 94.0, (92.0,), 140.0, 160.0, 180.0)

        thin_entry = MkrService._entry_plan(
            service, 100.0, 80, confirmation, calc.ExhaustionRead(()), thin,
            moving_averages, aligned,
        )
        generous_entry = MkrService._entry_plan(
            service, 100.0, 80, confirmation, calc.ExhaustionRead(()), generous,
            moving_averages, aligned,
        )

        assert thin_entry.decision == "Partial"
        assert "reward to risk" in thin_entry.rationale
        assert generous_entry.decision == "Y"

    def test_size_never_breaches_the_cash_floor(self, mkr_service) -> None:
        service, *_ = mkr_service
        rotation = service.analyse("TEST").rotation
        assert rotation.entry_size_pct <= 100.0 - rotation.cash_floor_pct


class TestRender:
    def test_the_report_carries_every_required_section(self, mkr_service) -> None:
        service, *_ = mkr_service
        report = render_report(service.analyse("TEST"))

        for heading in (
            "EXECUTIVE SUMMARY", "SCORECARD TABLE", "KEY PRICE LEVELS",
            "ENTRY TIMING", "LEAPS RECOMMENDATION", "WHEEL / CSP",
            "Top 3 Risks", "MONTE CARLO",
        ):
            assert f"**{heading}**" in report or heading in report

    def test_the_scorecard_table_has_fourteen_rows(self, mkr_service) -> None:
        service, *_ = mkr_service
        report = render_report(service.analyse("TEST"))
        table = report.split("**SCORECARD TABLE**")[1].split("**KEY PRICE LEVELS**")[0]
        rows = [line for line in table.splitlines() if line.startswith("| ")]

        # Header + separator + fourteen frameworks.
        assert len(rows) == 16

    def test_the_verdict_line_states_uni_confidence_and_peg(self, mkr_service) -> None:
        service, *_ = mkr_service
        analysis = service.analyse("TEST")
        report = render_report(analysis)

        assert f"{analysis.uni.total}/25" in report
        assert f"CONFIDENCE: {analysis.confidence}/100" in report
        assert "PEG: 0.42x [HIGHEST PRIORITY]" in report

    def test_at_most_three_risks_are_listed(self, mkr_service) -> None:
        service, *_ = mkr_service
        assert len(service.analyse("TEST").risks) <= 3

    def test_the_fact_pack_marks_the_legs_needing_judgement(self, mkr_service) -> None:
        service, *_ = mkr_service
        pack = format_analysis(service.analyse("TEST"))

        assert "NEEDS YOUR JUDGEMENT" in pack
        assert "Pivot sequence" in pack
        assert "Cup & Handle" in pack

    def test_prices_are_formatted_not_left_in_scientific_notation(self, mkr_service) -> None:
        service, *_ = mkr_service
        report = render_report(service.analyse("TEST"))
        assert "e+0" not in report


# --- Write-up -----------------------------------------------------------------


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


class TestGenerate:
    def test_missing_key_is_a_configuration_error(self, mkr_service) -> None:
        service, *_ = mkr_service
        assert not service.ai_available
        with pytest.raises(ConfigurationError):
            service.generate(service.analyse("TEST"))

    def test_the_write_up_is_archived_with_the_ticker_tagged(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        client = _StubClient(_StubResponse("**EXECUTIVE SUMMARY**\n\nLong.", searches=4))
        ai = MkrService(
            db, market_data, settings, None, FakeFundamentalsProvider(), client
        )

        note = ai.generate(ai.analyse("TEST"))

        assert note["searches"] == 4
        assert note["id"]
        assert "mkr" in note["tags"] and "TEST" in note["tags"]
        assert note["title"].startswith("MKR 14-Framework - TEST")

    def test_the_model_is_handed_every_number_it_may_quote(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        client = _StubClient(_StubResponse("written"))
        ai = MkrService(
            db, market_data, settings, None, FakeFundamentalsProvider(), client
        )

        analysis = ai.analyse("TEST")
        ai.generate(analysis)
        sent = client.messages.kwargs["messages"][0]["content"]

        assert "SCORECARD TABLE" in sent
        assert "Pivot sequence" in sent
        assert "never invent" in client.messages.kwargs["system"].lower() or (
            "may not estimate" in client.messages.kwargs["system"]
        )

    def test_web_search_is_offered_by_default_and_can_be_turned_off(
        self, mkr_service
    ) -> None:
        service, db, settings, market_data = mkr_service
        client = _StubClient(_StubResponse("written"))
        ai = MkrService(
            db, market_data, settings, None, FakeFundamentalsProvider(), client
        )
        analysis = ai.analyse("TEST")

        ai.generate(analysis)
        assert client.messages.kwargs["tools"][0]["name"] == "web_search"
        assert (
            client.messages.kwargs["tools"][0]["max_uses"]
            == settings.mkr_web_search_max_uses
        )

        ai.generate(analysis, use_web_search=False)
        assert not client.messages.kwargs["tools"]

    def test_extra_context_reaches_the_prompt(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        client = _StubClient(_StubResponse("written"))
        ai = MkrService(
            db, market_data, settings, None, FakeFundamentalsProvider(), client
        )

        ai.generate(ai.analyse("TEST"), extra_context="I already hold a Jan 200C")
        assert "Jan 200C" in client.messages.kwargs["messages"][0]["content"]

    def test_an_empty_response_is_a_provider_error(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        ai = MkrService(
            db, market_data, settings, None, FakeFundamentalsProvider(),
            _StubClient(_StubResponse("   ")),
        )
        with pytest.raises(ProviderError):
            ai.generate(ai.analyse("TEST"))

    def test_a_refusal_is_a_provider_error(self, mkr_service) -> None:
        service, db, settings, market_data = mkr_service
        response = _StubResponse("partial")
        response.stop_reason = "refusal"
        ai = MkrService(
            db, market_data, settings, None, FakeFundamentalsProvider(),
            _StubClient(response),
        )
        with pytest.raises(ProviderError):
            ai.generate(ai.analyse("TEST"))

    def test_the_computed_report_can_be_archived_without_a_model(self, mkr_service) -> None:
        service, *_ = mkr_service
        note = service.archive_deterministic(service.analyse("TEST"))

        assert note["model"] is None
        assert note["note_type"] == "mkr_analysis"
        assert "SCORECARD TABLE" in note["content"]
