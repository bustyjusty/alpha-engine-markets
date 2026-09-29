"""MKR 14-Framework analysis — the deterministic half.

Everything in this module is pure: OHLCV frames in, numbers and labelled
signals out. No I/O, no providers, no model calls. The service layer
(:mod:`market_intel.services.mkr`) adds the data it cannot compute from price
alone — fundamentals, the option chain — and optionally hands the whole
computed pack to Claude for the judgement layer.

The split matters. Six of the fourteen frameworks (Fibonacci, RSI, fair value
gaps, order flow, MA structure, momentum) are arithmetic and should never be
guessed by a language model; three more (multi-timeframe, technical
confirmation, capital rotation) are rule systems that are only useful if they
run identically every time. Those nine are computed here. Elliott Wave and
chart patterns are labelled by heuristic and explicitly marked ambiguous when
the evidence is thin, because a confident wave count from a machine that cannot
see the chart is worse than no count at all.

Conventions:

* Frames are date-indexed with lowercase ``open/high/low/close/volume`` columns
  — the shape :meth:`MarketDataService.get_price_history` returns.
* Every read degrades to ``None``/"insufficient history" rather than raising,
  so a six-month-old listing still produces a report with honest gaps.
* ``Signal`` is the shared vocabulary for the scorecard: one of bullish,
  bearish, neutral, or ``n/a`` when the framework could not be evaluated.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from market_intel.analysis.indicators import bollinger, ema, macd, rolling_vwap, rsi, sma

Signal = Literal["bullish", "bearish", "neutral", "n/a"]
Strength = Literal["strong", "moderate", "weak", "—"]

#: Fibonacci retracement ratios, in the order traders read them.
FIB_RATIOS: tuple[float, ...] = (0.236, 0.382, 0.5, 0.618, 0.786)
#: The "golden pocket" — the 61.8-65% band where trend entries cluster.
GOLDEN_POCKET: tuple[float, float] = (0.618, 0.65)
#: Extension multiples projected beyond the impulse.
FIB_EXTENSIONS: tuple[float, ...] = (1.272, 1.618, 2.618)

RSI_EXHAUSTION = 78.0
VOLUME_CLIMAX_MULTIPLE = 2.0
ADX_TRENDING = 25.0
#: Stop distance in ATR multiples for position sizing.
ATR_STOP_MULTIPLE = 2.0


# --- Swing structure ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pivot:
    """A confirmed swing point.

    ``kind`` is "high" or "low". A pivot is only confirmed once ``right``
    bars have printed after it, so the most recent bars never produce one —
    that lag is the price of not repainting.
    """

    date: dt.date
    price: float
    kind: str

    @property
    def is_high(self) -> bool:
        return self.kind == "high"


def find_pivots(frame: pd.DataFrame, left: int = 4, right: int = 4) -> list[Pivot]:
    """Fractal swing highs and lows, oldest first.

    A bar is a swing high when its high is the highest of the window spanning
    ``left`` bars before and ``right`` bars after it. Consecutive pivots of the
    same kind are collapsed to the more extreme one, so the output always
    alternates high/low and reads as a swing sequence.
    """
    if frame.empty or len(frame) < left + right + 1:
        return []

    high = _column(frame, "high")
    low = _column(frame, "low")
    raw: list[Pivot] = []
    for position in range(left, len(frame) - right):
        window = slice(position - left, position + right + 1)
        stamp = _as_date(frame.index[position])
        high_value = float(high.iloc[position])
        low_value = float(low.iloc[position])
        if high_value >= float(high.iloc[window].max()):
            raw.append(Pivot(stamp, high_value, "high"))
        elif low_value <= float(low.iloc[window].min()):
            raw.append(Pivot(stamp, low_value, "low"))

    collapsed: list[Pivot] = []
    for pivot in raw:
        if collapsed and collapsed[-1].kind == pivot.kind:
            previous = collapsed[-1]
            better = (
                pivot.price > previous.price
                if pivot.is_high
                else pivot.price < previous.price
            )
            if better:
                collapsed[-1] = pivot
            continue
        collapsed.append(pivot)
    return collapsed


@dataclass(frozen=True, slots=True)
class Swing:
    """The leg between two opposite pivots."""

    start: Pivot
    end: Pivot

    @property
    def is_up(self) -> bool:
        return self.end.price > self.start.price

    @property
    def low(self) -> float:
        return min(self.start.price, self.end.price)

    @property
    def high(self) -> float:
        return max(self.start.price, self.end.price)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def pct(self) -> float:
        base = self.start.price
        return (self.end.price / base - 1.0) * 100.0 if base else 0.0


def latest_swing(pivots: list[Pivot]) -> Swing | None:
    """The most recent completed leg, or None when there aren't two pivots."""
    if len(pivots) < 2:
        return None
    return Swing(pivots[-2], pivots[-1])


def impulse_swing(pivots: list[Pivot]) -> Swing | None:
    """The leg the Fibonacci grid should be drawn on.

    Traders measure retracements against the *impulse*, not against the
    correction that follows it. When the final leg is the smaller of the last
    two, it is the correction and the leg before it is the impulse — drawing
    the grid on the wrong one inverts every level, which is the kind of bug
    that still looks plausible on a chart.
    """
    if len(pivots) < 2:
        return None
    last = Swing(pivots[-2], pivots[-1])
    if len(pivots) >= 3:
        prior = Swing(pivots[-3], pivots[-2])
        if prior.range > last.range:
            return prior
    return last


# --- 1. Elliott Wave ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaveRead:
    """A candidate Elliott count derived from the swing sequence.

    This is a *labelling heuristic*, not a wave analyst. It counts alternating
    pivots from the last significant origin and reports what that implies,
    marking ``ambiguous`` whenever the sequence is too short, overlapping, or
    inconsistent with impulse rules. The service passes ``pivots`` to the model
    so the judgement layer can overrule this with the actual chart in mind.
    """

    position: str
    degree_origin: dt.date | None
    invalidation: float | None
    w3_extended: bool | None
    ambiguous: bool
    reason: str
    pivots: tuple[Pivot, ...] = ()

    def to_signal(self) -> FrameworkSignal:
        if self.ambiguous:
            return FrameworkSignal(
                "Elliott Wave", "neutral", "weak", f"{self.position} (ambiguous)"
            )
        impulse_up = self.position.startswith("W") and self.position not in ("W4",)
        signal: Signal = "bullish" if impulse_up else "bearish"
        if self.position in ("W4", "B"):
            signal = "neutral"
        strength: Strength = "moderate" if self.w3_extended is None else "strong"
        return FrameworkSignal("Elliott Wave", signal, strength, self.position)


def wave_read(pivots: list[Pivot], last_close: float) -> WaveRead:
    """Label the current wave position from the alternating pivot sequence.

    The count is deliberately conservative: it uses the last five pivots, tests
    the two rules that are objectively checkable (wave 3 is never the shortest,
    wave 4 does not overlap wave 1), and declares ambiguity when either fails.
    """
    if len(pivots) < 4:
        return WaveRead(
            "unlabelled", None, None, None, True,
            "fewer than four confirmed pivots — no countable structure",
            tuple(pivots),
        )

    recent = pivots[-5:]
    origin = recent[0]
    legs = [
        abs(recent[index + 1].price - recent[index].price)
        for index in range(len(recent) - 1)
    ]
    upward = recent[-1].price > origin.price

    # Wave-3-is-never-shortest is only testable once three legs exist.
    w3_extended: bool | None = None
    ambiguous = False
    reason = "swing sequence is consistent with an impulse"
    if len(legs) >= 3:
        w1, w2, w3 = legs[0], legs[1], legs[2]
        if w3 < w1:
            ambiguous = True
            reason = "third leg is shorter than the first — impulse rule broken"
        else:
            w3_extended = w3 > w1 * 1.618
        # Wave 4 must not trade into wave 1 territory.
        if len(recent) >= 5 and upward:
            wave_one_top = recent[1].price
            wave_four_low = recent[4].price if not recent[4].is_high else None
            if wave_four_low is not None and wave_four_low < wave_one_top:
                ambiguous = True
                reason = "fourth leg overlaps the first — corrective, not impulsive"
        _ = w2

    last_pivot = recent[-1]
    if upward:
        position = "W5 (impulse up)" if last_pivot.is_high else "W4 (correction)"
    else:
        position = "C (corrective down)" if not last_pivot.is_high else "B (bounce)"
    if last_close > max(pivot.price for pivot in recent):
        position = "W5 extension above prior high"

    invalidation = min(
        (pivot.price for pivot in recent if not pivot.is_high), default=None
    )
    return WaveRead(
        position, origin.date, invalidation, w3_extended, ambiguous, reason,
        tuple(recent),
    )


# --- 2. Fibonacci -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FibRead:
    """Retracement grid and extension targets for the latest impulse."""

    swing: Swing | None
    retracements: dict[str, float] = field(default_factory=dict)
    golden_pocket: tuple[float, float] | None = None
    extensions: dict[str, float] = field(default_factory=dict)
    extension_anchor: str = ""
    position_pct: float | None = None  # where price sits in the swing, 0-100
    retraced_pct: float | None = None  # how much of the impulse has been given back
    in_golden_pocket: bool = False

    def to_signal(self) -> FrameworkSignal:
        if self.swing is None or self.retraced_pct is None:
            return FrameworkSignal("Fibonacci", "n/a", "—", "no confirmed swing")
        if self.in_golden_pocket:
            return FrameworkSignal(
                "Fibonacci", "bullish", "strong", "price in the golden pocket"
            )
        if self.retraced_pct < 0.0:
            return FrameworkSignal(
                "Fibonacci", "bullish", "moderate",
                f"{-self.retraced_pct:.0f}% beyond the swing — extension",
            )
        if self.retraced_pct > 78.6:
            return FrameworkSignal(
                "Fibonacci", "bearish", "moderate",
                f"{self.retraced_pct:.0f}% retraced — past the 78.6%",
            )
        return FrameworkSignal(
            "Fibonacci", "neutral", "weak", f"{self.retraced_pct:.0f}% retraced"
        )


def fib_read(pivots: list[Pivot], last_close: float) -> FibRead:
    """Build the retracement grid and extension targets.

    Retracements are measured against the direction of the impulse: an up-leg
    retraces downward from the high, a down-leg retraces upward from the low.
    Extensions project from the correction that followed the impulse (the
    classic A-B-C anchor) when three pivots are available, and from the impulse
    origin otherwise — ``extension_anchor`` says which was used, and both
    project in the direction of the impulse rather than the correction.
    """
    swing = impulse_swing(pivots)
    if swing is None or swing.range <= 0:
        return FibRead(None)

    span = swing.range
    retracements: dict[str, float] = {}
    for ratio in FIB_RATIOS:
        level = swing.high - span * ratio if swing.is_up else swing.low + span * ratio
        retracements[f"{ratio * 100:.1f}%"] = level

    if swing.is_up:
        pocket_pair = (
            swing.high - span * GOLDEN_POCKET[1],
            swing.high - span * GOLDEN_POCKET[0],
        )
    else:
        pocket_pair = (
            swing.low + span * GOLDEN_POCKET[0],
            swing.low + span * GOLDEN_POCKET[1],
        )
    pocket = (min(pocket_pair), max(pocket_pair))

    # Anchor the extensions on the correction when there is one, and always
    # project in the direction the impulse was travelling.
    correction_end = pivots[-1] if pivots[-1] is not swing.end else None
    if correction_end is not None:
        base = correction_end.price
        anchor_label = f"last correction at {base:,.2f}"
    else:
        base = swing.low if swing.is_up else swing.high
        anchor_label = f"impulse origin at {base:,.2f}"
    direction = 1.0 if swing.is_up else -1.0
    extensions = {
        f"{multiple:.3f}x": base + direction * span * multiple
        for multiple in FIB_EXTENSIONS
    }

    position = (last_close - swing.low) / span * 100.0
    retraced = (
        (swing.high - last_close) / span * 100.0
        if swing.is_up
        else (last_close - swing.low) / span * 100.0
    )
    return FibRead(
        swing, retracements, pocket, extensions, anchor_label, position, retraced,
        pocket[0] <= last_close <= pocket[1],
    )


# --- 3. Chart patterns --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatternCandidate:
    """One pattern the price series is consistent with."""

    name: str
    state: str  # "forming" | "confirmed"
    target: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class PatternRead:
    """Deterministically detectable patterns only.

    Flags, coils and channels fall out of range arithmetic, so they are
    detected here. Cup & handle and head & shoulders need shape recognition
    that a rule on daily closes gets wrong more often than right — those are
    left to the judgement layer, and ``model_checks`` says so explicitly.
    """

    candidates: tuple[PatternCandidate, ...] = ()
    model_checks: tuple[str, ...] = ("Cup & Handle", "Head & Shoulders")

    def to_signal(self) -> FrameworkSignal:
        if not self.candidates:
            return FrameworkSignal("Chart Patterns", "neutral", "weak", "no clean setup")
        primary = self.candidates[0]
        signal: Signal = "neutral"
        lowered = primary.name.lower()
        if "bull" in lowered or "ascending" in lowered:
            signal = "bullish"
        elif "bear" in lowered or "descending" in lowered:
            signal = "bearish"
        strength: Strength = "strong" if primary.state == "confirmed" else "moderate"
        return FrameworkSignal(
            "Chart Patterns", signal, strength, f"{primary.name} ({primary.state})"
        )


def pattern_read(frame: pd.DataFrame, pivots: list[Pivot]) -> PatternRead:
    """Detect flags, coils and triangles from range behaviour."""
    if len(frame) < 60:
        return PatternRead()

    close = _column(frame, "close")
    high = _column(frame, "high")
    low = _column(frame, "low")
    last = float(close.iloc[-1])

    candidates: list[PatternCandidate] = []

    # Flag: a sharp impulse followed by a shallow, narrowing drift against it.
    impulse = float(close.iloc[-30]) / float(close.iloc[-60]) - 1.0
    drift = last / float(close.iloc[-30]) - 1.0
    recent_range = float(high.iloc[-20:].max() - low.iloc[-20:].min())
    prior_range = float(high.iloc[-60:-20].max() - low.iloc[-60:-20].min())
    contracting = prior_range > 0 and recent_range < prior_range * 0.6
    if impulse > 0.12 and -0.10 < drift <= 0.03 and contracting:
        pole = float(close.iloc[-30]) - float(close.iloc[-60])
        candidates.append(
            PatternCandidate(
                "Bull Flag",
                "confirmed" if last > float(high.iloc[-20:-1].max()) else "forming",
                last + pole,
                f"{impulse * 100:.0f}% pole, {drift * 100:+.1f}% drift, range halved",
            )
        )
    elif impulse < -0.12 and -0.03 <= drift < 0.10 and contracting:
        pole = float(close.iloc[-30]) - float(close.iloc[-60])
        candidates.append(
            PatternCandidate(
                "Bear Flag",
                "confirmed" if last < float(low.iloc[-20:-1].min()) else "forming",
                last + pole,
                f"{impulse * 100:.0f}% pole, {drift * 100:+.1f}% drift, range halved",
            )
        )

    # Coil: Bollinger bandwidth in the bottom decile of the last year.
    upper, middle, lower = bollinger(close)
    bandwidth = ((upper - lower) / middle).dropna()
    if len(bandwidth) >= 60:
        current_width = float(bandwidth.iloc[-1])
        percentile = float((bandwidth.iloc[-252:] < current_width).mean() * 100.0)
        if percentile <= 10.0:
            candidates.append(
                PatternCandidate(
                    "Coil / volatility squeeze",
                    "forming",
                    None,
                    f"bandwidth in the {percentile:.0f}th percentile of the year",
                )
            )

    # Triangle: converging swing highs and lows.
    highs = [pivot for pivot in pivots if pivot.is_high][-3:]
    lows = [pivot for pivot in pivots if not pivot.is_high][-3:]
    if len(highs) >= 2 and len(lows) >= 2:
        falling_tops = highs[-1].price < highs[-2].price
        rising_bottoms = lows[-1].price > lows[-2].price
        if falling_tops and rising_bottoms:
            apex = (highs[-1].price + lows[-1].price) / 2.0
            height = highs[-2].price - lows[-2].price
            candidates.append(
                PatternCandidate(
                    "Symmetrical Triangle", "forming", apex + height,
                    "lower highs into higher lows",
                )
            )
        elif rising_bottoms and not falling_tops:
            candidates.append(
                PatternCandidate(
                    "Ascending Triangle", "forming",
                    highs[-1].price + (highs[-1].price - lows[-2].price),
                    f"flat resistance at {highs[-1].price:,.2f}, higher lows",
                )
            )
        elif falling_tops and not rising_bottoms:
            candidates.append(
                PatternCandidate(
                    "Descending Triangle", "forming",
                    lows[-1].price - (highs[-2].price - lows[-1].price),
                    f"flat support at {lows[-1].price:,.2f}, lower highs",
                )
            )

    return PatternRead(tuple(candidates))


# --- 4. RSI -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RsiRead:
    """Daily and weekly RSI plus divergence state."""

    daily: float | None
    weekly: float | None
    divergence: str | None  # e.g. "regular bearish"
    divergence_detail: str = ""

    def to_signal(self) -> FrameworkSignal:
        if self.daily is None:
            return FrameworkSignal("RSI", "n/a", "—", "insufficient history")
        detail = f"D {self.daily:.0f}"
        if self.weekly is not None:
            detail += f" / W {self.weekly:.0f}"
        if self.divergence:
            detail += f" · {self.divergence} divergence"
            if self.divergence.startswith("regular bearish"):
                return FrameworkSignal("RSI", "bearish", "strong", detail)
            if self.divergence.startswith("regular bullish"):
                return FrameworkSignal("RSI", "bullish", "strong", detail)
            # Hidden divergence is a continuation signal, not a reversal one.
            if "hidden bullish" in self.divergence:
                return FrameworkSignal("RSI", "bullish", "moderate", detail)
            if "hidden bearish" in self.divergence:
                return FrameworkSignal("RSI", "bearish", "moderate", detail)
        if self.daily >= 70.0:
            return FrameworkSignal("RSI", "bearish", "moderate", detail + " · overbought")
        if self.daily <= 30.0:
            return FrameworkSignal("RSI", "bullish", "moderate", detail + " · oversold")
        return FrameworkSignal(
            "RSI", "bullish" if self.daily > 50.0 else "bearish", "weak", detail
        )


def rsi_read(frame: pd.DataFrame, weekly: pd.DataFrame | None = None) -> RsiRead:
    """RSI on both timeframes, with divergence against the last two swings."""
    if frame.empty:
        return RsiRead(None, None, None)
    close = _column(frame, "close")
    daily_series = rsi(close)
    daily = _last_float(daily_series)
    weekly_value = None
    if weekly is not None and len(weekly) >= 15:
        weekly_value = _last_float(rsi(_column(weekly, "close")))

    divergence, detail = _rsi_divergence(frame, daily_series)
    return RsiRead(daily, weekly_value, divergence, detail)


def _rsi_divergence(
    frame: pd.DataFrame, rsi_series: pd.Series
) -> tuple[str | None, str]:
    """Compare the last two price swings against RSI at the same dates."""
    pivots = find_pivots(frame, left=3, right=3)
    highs = [pivot for pivot in pivots if pivot.is_high][-2:]
    lows = [pivot for pivot in pivots if not pivot.is_high][-2:]

    def rsi_at(stamp: dt.date) -> float | None:
        try:
            value = rsi_series.loc[stamp]
        except KeyError:
            return None
        return None if pd.isna(value) else float(value)

    if len(highs) == 2:
        first, second = highs
        r1, r2 = rsi_at(first.date), rsi_at(second.date)
        if r1 is not None and r2 is not None:
            if second.price > first.price and r2 < r1:
                return (
                    "regular bearish",
                    f"price {first.price:,.2f}→{second.price:,.2f} but RSI {r1:.0f}→{r2:.0f}",
                )
            if second.price < first.price and r2 > r1:
                return (
                    "hidden bearish",
                    f"lower high {second.price:,.2f} on stronger RSI {r2:.0f}",
                )
    if len(lows) == 2:
        first, second = lows
        r1, r2 = rsi_at(first.date), rsi_at(second.date)
        if r1 is not None and r2 is not None:
            if second.price < first.price and r2 > r1:
                return (
                    "regular bullish",
                    f"price {first.price:,.2f}→{second.price:,.2f} but RSI {r1:.0f}→{r2:.0f}",
                )
            if second.price > first.price and r2 < r1:
                return (
                    "hidden bullish",
                    f"higher low {second.price:,.2f} on cooler RSI {r2:.0f}",
                )
    return None, ""


# --- 5. Fair value gaps -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FairValueGap:
    """A three-bar imbalance that price has not traded back through."""

    date: dt.date
    low: float
    high: float
    direction: str  # "bullish" | "bearish"
    timeframe: str

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True, slots=True)
class FvgRead:
    """Unfilled gaps either side of spot."""

    unfilled: tuple[FairValueGap, ...] = ()
    nearest_above: FairValueGap | None = None
    nearest_below: FairValueGap | None = None
    intraday_available: bool = False
    last_close: float | None = None

    def to_signal(self) -> FrameworkSignal:
        """Direction comes from whichever draw is closer to spot.

        An unfilled gap is a magnet in both directions — calling one "support"
        and the other "resistance" is the mistake. What matters is which side
        price has to travel less far to fill.
        """
        if not self.unfilled:
            return FrameworkSignal("Fair Value Gaps", "neutral", "weak", "none unfilled")
        above, below = self.nearest_above, self.nearest_below
        detail_parts = []
        if below:
            detail_parts.append(f"draw below {below.low:,.2f}-{below.high:,.2f}")
        if above:
            detail_parts.append(f"draw above {above.low:,.2f}-{above.high:,.2f}")
        detail = " · ".join(detail_parts) or f"{len(self.unfilled)} unfilled"

        if above and not below:
            return FrameworkSignal("Fair Value Gaps", "bullish", "moderate", detail)
        if below and not above:
            return FrameworkSignal("Fair Value Gaps", "bearish", "moderate", detail)
        if above and below and self.last_close:
            up_distance = above.midpoint - self.last_close
            down_distance = self.last_close - below.midpoint
            if up_distance < down_distance * 0.75:
                return FrameworkSignal("Fair Value Gaps", "bullish", "weak", detail)
            if down_distance < up_distance * 0.75:
                return FrameworkSignal("Fair Value Gaps", "bearish", "weak", detail)
        return FrameworkSignal("Fair Value Gaps", "neutral", "weak", detail)


def find_fair_value_gaps(
    frame: pd.DataFrame, timeframe: str = "daily", lookback: int = 120
) -> list[FairValueGap]:
    """Unfilled three-bar FVGs, newest last.

    Bullish gap: bar *i*'s low prints above bar *i-2*'s high, leaving the band
    between them untraded. A gap counts as filled once any later bar trades
    fully back through it.
    """
    if len(frame) < 3:
        return []
    window = frame.iloc[-lookback:] if lookback else frame
    high = _column(window, "high").to_numpy(dtype=float)
    low = _column(window, "low").to_numpy(dtype=float)
    dates = [_as_date(stamp) for stamp in window.index]

    gaps: list[FairValueGap] = []
    for index in range(2, len(window)):
        if low[index] > high[index - 2]:
            gaps.append(
                FairValueGap(
                    dates[index], high[index - 2], low[index], "bullish", timeframe
                )
            )
        elif high[index] < low[index - 2]:
            gaps.append(
                FairValueGap(
                    dates[index], high[index], low[index - 2], "bearish", timeframe
                )
            )

    unfilled: list[FairValueGap] = []
    for gap in gaps:
        after = window[[_as_date(stamp) > gap.date for stamp in window.index]]
        if after.empty:
            unfilled.append(gap)
            continue
        traded_through = (
            float(_column(after, "low").min()) <= gap.low
            if gap.direction == "bullish"
            else float(_column(after, "high").max()) >= gap.high
        )
        if not traded_through:
            unfilled.append(gap)
    return unfilled


def fvg_read(
    frame: pd.DataFrame, last_close: float, intraday: pd.DataFrame | None = None
) -> FvgRead:
    """Daily gaps, plus 4-hour gaps when intraday bars were available."""
    gaps = find_fair_value_gaps(frame, "daily")
    if intraday is not None and len(intraday) >= 3:
        gaps.extend(find_fair_value_gaps(intraday, "4H", lookback=180))
    if not gaps:
        return FvgRead(
            intraday_available=intraday is not None, last_close=last_close
        )

    above = [gap for gap in gaps if gap.low > last_close]
    below = [gap for gap in gaps if gap.high < last_close]
    return FvgRead(
        tuple(gaps),
        min(above, key=lambda gap: gap.low) if above else None,
        max(below, key=lambda gap: gap.high) if below else None,
        intraday is not None,
        last_close,
    )


# --- 6. Order flow & volume ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlowRead:
    """Volume trend, OBV direction and climax events."""

    last_volume: float | None
    average_20d: float | None
    ratio: float | None
    trend: str
    obv_direction: str
    climax_dates: tuple[dt.date, ...] = ()

    def to_signal(self) -> FrameworkSignal:
        if self.ratio is None:
            return FrameworkSignal("Order Flow & Volume", "n/a", "—", "no volume data")
        detail = f"{self.ratio:.2f}x 20d avg · OBV {self.obv_direction}"
        if self.obv_direction == "rising" and self.ratio >= 1.0:
            return FrameworkSignal("Order Flow & Volume", "bullish", "strong", detail)
        if self.obv_direction == "falling" and self.ratio >= 1.0:
            return FrameworkSignal("Order Flow & Volume", "bearish", "strong", detail)
        if self.obv_direction == "rising":
            return FrameworkSignal("Order Flow & Volume", "bullish", "weak", detail)
        if self.obv_direction == "falling":
            return FrameworkSignal("Order Flow & Volume", "bearish", "weak", detail)
        return FrameworkSignal("Order Flow & Volume", "neutral", "weak", detail)


def on_balance_volume(frame: pd.DataFrame) -> pd.Series:
    """Cumulative volume signed by the daily close direction."""
    close = _column(frame, "close")
    volume = _column(frame, "volume").fillna(0.0)
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def flow_read(frame: pd.DataFrame) -> FlowRead:
    """Volume versus its 20-day average, OBV slope, and climax bars."""
    if frame.empty or "volume" not in frame.columns:
        return FlowRead(None, None, None, "unknown", "unknown")
    volume = _column(frame, "volume").fillna(0.0)
    if float(volume.sum()) <= 0:
        return FlowRead(None, None, None, "unknown", "unknown")

    last = float(volume.iloc[-1])
    average = _last_float(volume.rolling(20, min_periods=5).mean())
    ratio = last / average if average else None

    recent_average = float(volume.iloc[-20:].mean())
    prior_average = float(volume.iloc[-60:-20].mean()) if len(volume) >= 60 else recent_average
    if prior_average > 0 and recent_average > prior_average * 1.15:
        trend = "expanding"
    elif prior_average > 0 and recent_average < prior_average * 0.85:
        trend = "contracting"
    else:
        trend = "steady"

    obv = on_balance_volume(frame)
    obv_direction = "flat"
    if len(obv) >= 21:
        change = float(obv.iloc[-1] - obv.iloc[-21])
        scale = float(volume.iloc[-20:].mean()) * 5.0
        if scale > 0 and change > scale:
            obv_direction = "rising"
        elif scale > 0 and change < -scale:
            obv_direction = "falling"

    climax: list[dt.date] = []
    if average:
        rolling_average = volume.rolling(20, min_periods=5).mean()
        for stamp, value, mean_value in zip(
            frame.index[-30:], volume.iloc[-30:], rolling_average.iloc[-30:]
        ):
            if mean_value and float(value) >= float(mean_value) * VOLUME_CLIMAX_MULTIPLE:
                climax.append(_as_date(stamp))

    return FlowRead(last, average, ratio, trend, obv_direction, tuple(climax))


# --- 7. MA structure and support/resistance -----------------------------------


@dataclass(frozen=True, slots=True)
class MaRead:
    """Moving-average stack, cross state and horizontal levels."""

    ema20: float | None
    ema50: float | None
    sma200: float | None
    ema200: float | None
    vwap20: float | None
    cross: str  # "golden cross" | "death cross" | "none"
    all_time_high: float | None
    supports: tuple[float, ...] = ()
    resistances: tuple[float, ...] = ()

    def to_signal(self) -> FrameworkSignal:
        if self.ema20 is None or self.ema50 is None:
            return FrameworkSignal("MA Structure + S/R", "n/a", "—", "insufficient history")
        stacked_up = (
            self.sma200 is not None
            and self.ema20 > self.ema50 > self.sma200
        )
        stacked_down = (
            self.sma200 is not None
            and self.ema20 < self.ema50 < self.sma200
        )
        detail = f"20 {self.ema20:,.2f} / 50 {self.ema50:,.2f}"
        if self.sma200 is not None:
            detail += f" / 200 {self.sma200:,.2f}"
        if self.cross != "none":
            detail += f" · {self.cross}"
        if stacked_up:
            return FrameworkSignal("MA Structure + S/R", "bullish", "strong", detail)
        if stacked_down:
            return FrameworkSignal("MA Structure + S/R", "bearish", "strong", detail)
        return FrameworkSignal(
            "MA Structure + S/R",
            "bullish" if self.ema20 > self.ema50 else "bearish",
            "moderate",
            detail,
        )


def ma_read(frame: pd.DataFrame, pivots: list[Pivot], last_close: float) -> MaRead:
    """Moving averages plus the horizontal levels that matter around spot."""
    if frame.empty:
        return MaRead(None, None, None, None, None, "none", None)
    close = _column(frame, "close")
    ema20 = _last_float(ema(close, 20))
    ema50 = _last_float(ema(close, 50))
    ema200 = _last_float(ema(close, 200)) if len(close) >= 200 else None
    sma200 = _last_float(sma(close, 200))
    vwap20 = _last_float(rolling_vwap(frame))

    cross = "none"
    if len(close) >= 210 and sma200 is not None:
        fast = ema(close, 50)
        slow = sma(close, 200)
        spread = (fast - slow).dropna()
        if len(spread) >= 40:
            current = float(spread.iloc[-1])
            past = float(spread.iloc[-40])
            if current > 0 and past <= 0:
                cross = "golden cross (last 40 sessions)"
            elif current < 0 and past >= 0:
                cross = "death cross (last 40 sessions)"
            elif current > 0:
                cross = "golden cross intact"
            else:
                cross = "death cross intact"

    all_time_high = float(_column(frame, "high").max())
    pivot_lows = sorted(
        {pivot.price for pivot in pivots if not pivot.is_high and pivot.price < last_close},
        reverse=True,
    )
    pivot_highs = sorted(
        {pivot.price for pivot in pivots if pivot.is_high and pivot.price > last_close}
    )
    return MaRead(
        ema20, ema50, sma200, ema200, vwap20, cross, all_time_high,
        tuple(pivot_lows[:3]), tuple(pivot_highs[:3]),
    )


# --- 8. Momentum --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MomentumRead:
    """MACD, ADX, Bollinger position and ATR."""

    macd_hist: float | None
    macd_state: str
    adx: float | None
    percent_b: float | None
    sigma: float | None  # standard deviations from the 20-day mean
    atr: float | None
    atr_pct: float | None

    @property
    def trending(self) -> bool:
        return self.adx is not None and self.adx >= ADX_TRENDING

    def to_signal(self) -> FrameworkSignal:
        if self.macd_hist is None:
            return FrameworkSignal("Momentum", "n/a", "—", "insufficient history")
        detail = f"MACD hist {self.macd_hist:+.3f} {self.macd_state}"
        if self.adx is not None:
            detail += f" · ADX {self.adx:.0f}"
        if self.percent_b is not None:
            detail += f" · %B {self.percent_b:.2f}"
        strength: Strength = "strong" if self.trending else "moderate"
        if self.macd_hist > 0 and self.macd_state in ("rising", "bullish cross"):
            return FrameworkSignal("Momentum", "bullish", strength, detail)
        if self.macd_hist < 0 and self.macd_state in ("falling", "bearish cross"):
            return FrameworkSignal("Momentum", "bearish", strength, detail)
        return FrameworkSignal("Momentum", "neutral", "weak", detail)


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    high = _column(frame, "high")
    low = _column(frame, "low")
    close = _column(frame, "close")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder). Values above 25 mean trending."""
    high = _column(frame, "high")
    low = _column(frame, "low")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_series = average_true_range(frame, period)
    smoothing = {"alpha": 1.0 / period, "min_periods": period, "adjust": False}
    plus_di = 100.0 * plus_dm.ewm(**smoothing).mean() / atr_series
    minus_di = 100.0 * minus_dm.ewm(**smoothing).mean() / atr_series
    total = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / total
    return dx.ewm(**smoothing).mean()


def momentum_read(frame: pd.DataFrame) -> MomentumRead:
    """Latest momentum picture across MACD, ADX, Bollinger and ATR."""
    if len(frame) < 35:
        return MomentumRead(None, "unknown", None, None, None, None, None)
    close = _column(frame, "close")
    histogram = macd(close)[2].dropna()
    hist_value = float(histogram.iloc[-1]) if len(histogram) else None
    state = "unknown"
    if len(histogram) >= 2:
        previous = float(histogram.iloc[-2])
        current = float(histogram.iloc[-1])
        if previous <= 0 < current:
            state = "bullish cross"
        elif previous >= 0 > current:
            state = "bearish cross"
        elif current > previous:
            state = "rising"
        elif current < previous:
            state = "falling"
        else:
            state = "flat"

    adx_value = _last_float(adx(frame))
    upper, middle, lower = bollinger(close)
    percent_b = None
    upper_value, lower_value, middle_value = (
        _last_float(upper), _last_float(lower), _last_float(middle)
    )
    last = float(close.iloc[-1])
    if upper_value is not None and lower_value is not None and upper_value != lower_value:
        percent_b = (last - lower_value) / (upper_value - lower_value)
    sigma = None
    if middle_value is not None and upper_value is not None:
        std = (upper_value - middle_value) / 2.0
        sigma = (last - middle_value) / std if std else None

    atr_value = _last_float(average_true_range(frame))
    atr_pct = atr_value / last * 100.0 if atr_value and last else None
    return MomentumRead(hist_value, state, adx_value, percent_b, sigma, atr_value, atr_pct)


# --- 11. Multi-timeframe ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimeframeTrend:
    """Trend verdict for one timeframe."""

    timeframe: str
    trend: Signal
    detail: str


@dataclass(frozen=True, slots=True)
class TimeframeRead:
    """Daily / weekly / monthly alignment."""

    trends: tuple[TimeframeTrend, ...] = ()

    @property
    def aligned(self) -> bool:
        usable = [trend.trend for trend in self.trends if trend.trend != "n/a"]
        return len(usable) >= 2 and len(set(usable)) == 1

    def to_signal(self) -> FrameworkSignal:
        if not self.trends:
            return FrameworkSignal("Multi-Timeframe", "n/a", "—", "insufficient history")
        detail = " · ".join(f"{t.timeframe} {t.trend}" for t in self.trends)
        if self.aligned:
            direction: Signal = next(t.trend for t in self.trends if t.trend != "n/a")
            return FrameworkSignal("Multi-Timeframe", direction, "strong", detail)
        return FrameworkSignal("Multi-Timeframe", "neutral", "weak", detail + " — mixed")


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily bars to a coarser timeframe (``W``, ``ME``, ``4h``)."""
    if frame.empty:
        return frame
    working = frame.copy()
    working.index = pd.to_datetime(working.index)
    aggregation: dict[str, str] = {}
    for column, how in (
        ("open", "first"), ("high", "max"), ("low", "min"),
        ("close", "last"), ("volume", "sum"),
    ):
        if column in working.columns:
            aggregation[column] = how
    resampled = working.resample(rule).agg(aggregation).dropna(subset=["close"])
    resampled.index = [stamp.date() for stamp in resampled.index]
    resampled.index.name = "date"
    return resampled


def timeframe_read(
    daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame
) -> TimeframeRead:
    """Trend on each timeframe from the fast/slow moving-average relationship."""
    specs = (
        ("Daily", daily, 20, 50),
        ("Weekly", weekly, 10, 30),
        ("Monthly", monthly, 6, 12),
    )
    trends: list[TimeframeTrend] = []
    for label, frame, fast_span, slow_span in specs:
        if len(frame) < slow_span + 2:
            trends.append(TimeframeTrend(label, "n/a", "insufficient history"))
            continue
        close = _column(frame, "close")
        fast = _last_float(ema(close, fast_span))
        slow = _last_float(ema(close, slow_span))
        last = float(close.iloc[-1])
        if fast is None or slow is None:
            trends.append(TimeframeTrend(label, "n/a", "insufficient history"))
            continue
        detail = f"close {last:,.2f} vs {fast_span}EMA {fast:,.2f}, {slow_span}EMA {slow:,.2f}"
        if last > fast > slow:
            trends.append(TimeframeTrend(label, "bullish", detail))
        elif last < fast < slow:
            trends.append(TimeframeTrend(label, "bearish", detail))
        else:
            trends.append(TimeframeTrend(label, "neutral", detail))
    return TimeframeRead(tuple(trends))


# --- 13. Technical indicator confirmation -------------------------------------


@dataclass(frozen=True, slots=True)
class ConfirmationRead:
    """The confirmation stack, including the death-cross override.

    ``death_cross_override`` is the hard rule from the framework: when the
    50 EMA is below the 200 EMA *and* price is below the 50, every bullish
    count is invalidated regardless of what the wave structure says. The
    service propagates this into the executive summary and the entry decision.
    """

    ema20: float | None
    ema50: float | None
    ema200: float | None
    rsi14: float | None
    macd_trend: str
    trix: float | None
    trix_slope: str
    death_cross_override: bool = False

    def to_signal(self) -> FrameworkSignal:
        if self.ema20 is None or self.ema50 is None:
            return FrameworkSignal(
                "Technical Confirmation", "n/a", "—", "insufficient history"
            )
        if self.death_cross_override:
            return FrameworkSignal(
                "Technical Confirmation", "bearish", "strong",
                "DEATH CROSS OVERRIDE — bullish counts invalidated",
            )
        detail = f"MACD {self.macd_trend} · TRIX {self.trix_slope}"
        stacked = self.ema200 is not None and self.ema20 > self.ema50 > self.ema200
        if stacked and self.trix_slope == "rising":
            return FrameworkSignal("Technical Confirmation", "bullish", "strong", detail)
        if stacked:
            return FrameworkSignal("Technical Confirmation", "bullish", "moderate", detail)
        if self.ema20 < self.ema50:
            return FrameworkSignal("Technical Confirmation", "bearish", "moderate", detail)
        return FrameworkSignal("Technical Confirmation", "neutral", "weak", detail)


def trix(close: pd.Series, span: int = 15) -> pd.Series:
    """TRIX: rate of change of a triple-smoothed EMA, in basis points."""
    smoothed = ema(ema(ema(close.astype(float), span), span), span)
    return smoothed.pct_change() * 10000.0


def confirmation_read(frame: pd.DataFrame) -> ConfirmationRead:
    """EMA stack, Wilder RSI, MACD trend, TRIX slope and the override."""
    if len(frame) < 55:
        return ConfirmationRead(None, None, None, None, "unknown", None, "unknown")
    close = _column(frame, "close")
    ema20 = _last_float(ema(close, 20))
    ema50 = _last_float(ema(close, 50))
    ema200 = _last_float(ema(close, 200)) if len(close) >= 200 else None
    rsi14 = _last_float(rsi(close))

    histogram = macd(close)[2].dropna()
    macd_trend = "flat"
    if len(histogram) >= 3:
        recent = histogram.iloc[-3:]
        if float(recent.iloc[-1]) > float(recent.iloc[0]):
            macd_trend = "rising"
        elif float(recent.iloc[-1]) < float(recent.iloc[0]):
            macd_trend = "falling"

    trix_series = trix(close).dropna()
    trix_value = float(trix_series.iloc[-1]) if len(trix_series) else None
    trix_slope = "unknown"
    if len(trix_series) >= 6:
        trix_slope = (
            "rising"
            if float(trix_series.iloc[-1]) > float(trix_series.iloc[-6])
            else "falling"
        )

    last = float(close.iloc[-1])
    override = bool(
        ema200 is not None
        and ema50 is not None
        and ema50 < ema200
        and last < ema50
    )
    return ConfirmationRead(
        ema20, ema50, ema200, rsi14, macd_trend, trix_value, trix_slope, override
    )


# --- Exhaustion override ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExhaustionRead:
    """The wave-top gate: two or more flags before a top may be called."""

    flags: tuple[str, ...] = ()

    @property
    def triggered(self) -> bool:
        return len(self.flags) >= 2

    @property
    def summary(self) -> str:
        if not self.flags:
            return "no exhaustion flags"
        state = "TRIGGERED" if self.triggered else "watching"
        return f"{state} ({len(self.flags)}/2): " + ", ".join(self.flags)


def exhaustion_read(
    momentum: MomentumRead, rsi_state: RsiRead, flow: FlowRead
) -> ExhaustionRead:
    """Collect the four exhaustion flags the framework recognises."""
    flags: list[str] = []
    if rsi_state.daily is not None and rsi_state.daily > RSI_EXHAUSTION:
        flags.append(f"RSI {rsi_state.daily:.0f} > {RSI_EXHAUSTION:.0f}")
    if flow.ratio is not None and flow.ratio >= VOLUME_CLIMAX_MULTIPLE:
        flags.append(f"volume climax {flow.ratio:.1f}x")
    if rsi_state.divergence == "regular bearish":
        flags.append("negative divergence")
    if momentum.sigma is not None and momentum.sigma > 2.0:
        flags.append(f"{momentum.sigma:.1f}σ above the 20-day mean")
    return ExhaustionRead(tuple(flags))


# --- Monte Carlo --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Terminal-distribution statistics for one simulated regime."""

    label: str
    paths: int
    horizon_days: int
    win_rate: float
    average_win: float
    average_loss: float
    worst_1pct: float
    median_return: float
    percentile_5: float
    percentile_95: float
    annual_drift: float
    annual_vol: float


def monte_carlo(
    closes: pd.Series,
    label: str,
    horizon_days: int = 90,
    paths: int = 1000,
    annual_drift: float | None = None,
    lookback: int = 252,
    seed: int = 20260928,
) -> MonteCarloResult | None:
    """Bootstrap terminal returns over ``horizon_days``.

    Daily log returns are resampled with replacement from the trailing
    ``lookback`` sessions, which preserves the security's own fat tails instead
    of assuming normality. When ``annual_drift`` is given the sampled returns
    are recentred on it — that is what makes the "thesis-tilted" run different
    from the neutral one: same volatility, different expected path.
    """
    series = closes.astype(float).dropna()
    if len(series) < 40:
        return None
    log_returns = np.diff(np.log(series.to_numpy()))[-lookback:]
    if log_returns.size < 30:
        return None

    daily_vol = float(np.std(log_returns, ddof=1))
    observed_drift = float(np.mean(log_returns))
    if annual_drift is not None:
        target_daily = annual_drift / 252.0
        log_returns = log_returns - observed_drift + target_daily
        drift_used = annual_drift
    else:
        drift_used = observed_drift * 252.0

    generator = np.random.default_rng(seed)
    draws = generator.choice(log_returns, size=(paths, horizon_days), replace=True)
    terminal = np.exp(draws.sum(axis=1)) - 1.0

    wins = terminal[terminal > 0]
    losses = terminal[terminal <= 0]
    return MonteCarloResult(
        label=label,
        paths=paths,
        horizon_days=horizon_days,
        win_rate=float(wins.size / terminal.size * 100.0),
        average_win=float(wins.mean() * 100.0) if wins.size else 0.0,
        average_loss=float(losses.mean() * 100.0) if losses.size else 0.0,
        worst_1pct=float(np.percentile(terminal, 1) * 100.0),
        median_return=float(np.percentile(terminal, 50) * 100.0),
        percentile_5=float(np.percentile(terminal, 5) * 100.0),
        percentile_95=float(np.percentile(terminal, 95) * 100.0),
        annual_drift=float(drift_used * 100.0),
        annual_vol=float(daily_vol * math.sqrt(252) * 100.0),
    )


# --- Black-Scholes (for the options frameworks) -------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(
    spot: float, strike: float, years: float, vol: float, rate: float, kind: str = "call"
) -> float | None:
    """Black-Scholes delta. Returns None when the inputs cannot support one."""
    if spot <= 0 or strike <= 0 or years <= 0 or vol <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (
        vol * math.sqrt(years)
    )
    return norm_cdf(d1) if kind == "call" else norm_cdf(d1) - 1.0


def bs_price(
    spot: float, strike: float, years: float, vol: float, rate: float, kind: str = "call"
) -> float | None:
    """Black-Scholes premium for a European option."""
    if spot <= 0 or strike <= 0 or years <= 0 or vol <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / (
        vol * math.sqrt(years)
    )
    d2 = d1 - vol * math.sqrt(years)
    discount = math.exp(-rate * years)
    if kind == "call":
        return spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    return strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)


# --- Scorecard ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameworkSignal:
    """One row of the scorecard."""

    name: str
    signal: Signal
    strength: Strength
    detail: str = ""

    @property
    def score(self) -> int:
        """+2/+1/0/-1/-2 depending on direction and strength."""
        weight = {"strong": 2, "moderate": 1, "weak": 1, "—": 0}[self.strength]
        if self.signal == "bullish":
            return weight
        if self.signal == "bearish":
            return -weight
        return 0


def confidence_score(signals: list[FrameworkSignal], override: bool = False) -> int:
    """Map the scorecard onto 0-100.

    50 is "no edge either way". The scale is symmetric, so a fully bearish
    board scores near zero rather than being clipped at 50 — the number is a
    directional conviction reading, not a probability. The death-cross override
    caps it at 35 no matter what the rest of the board says.
    """
    scored = [signal for signal in signals if signal.signal != "n/a"]
    if not scored:
        return 50
    total = sum(signal.score for signal in scored)
    maximum = sum(2 for _ in scored)
    raw = 50.0 + (total / maximum) * 50.0 if maximum else 50.0
    value = int(round(max(0.0, min(100.0, raw))))
    return min(value, 35) if override else value


# --- Helpers ------------------------------------------------------------------


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a float column, falling back to close when H/L/V are absent."""
    if name in frame.columns:
        series = frame[name].astype(float)
        if name in ("high", "low") and series.isna().any():
            return series.fillna(frame["close"].astype(float))
        return series
    if name in ("high", "low", "open"):
        return frame["close"].astype(float)
    return pd.Series(float("nan"), index=frame.index)


def _last_float(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _as_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()
