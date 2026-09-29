"""MKR 14-Framework analysis: one ticker in, a structured trade plan out.

This is the single-name counterpart to the cross-asset recap pipeline, and it
follows the same design rule: **compute everything computable, and only then
ask a model to write.**

Three stages:

1. **Gather** — daily bars (three years, for the 200-day and the monthly
   timeframe), optional intraday bars for 4-hour gaps, optional fundamentals
   and option chain. Each optional feed can fail independently; the analysis
   records what was unavailable instead of guessing.
2. **Analyse** — run all fourteen frameworks. Nine are pure price arithmetic
   (:mod:`market_intel.analysis.mkr`), three come from the option chain and
   fundamentals, and two — the UNI score and capital rotation — are rule
   systems applied to the output of the first twelve. The result is an
   :class:`MkrAnalysis`, which already renders as the full report on its own.
3. **Write** — optionally hand the finished numbers to Claude with web search
   so the judgement layer (wave labelling in context, pattern confirmation,
   the actual catalyst calendar, the risk narrative) is researched rather than
   invented. The model is given every number it is allowed to quote and told
   it may not produce another.

Without an API key stage 3 is skipped and the deterministic render is the
report — the same contract as the recap pipeline, for the same reason: a trade
plan that only exists when a key is configured is not a trade plan.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
import numpy as np
import pandas as pd

from market_intel.analysis import mkr as calc
from market_intel.analysis.mkr import (
    ATR_STOP_MULTIPLE,
    ConfirmationRead,
    ExhaustionRead,
    FibRead,
    FlowRead,
    FrameworkSignal,
    FvgRead,
    MaRead,
    MomentumRead,
    MonteCarloResult,
    PatternRead,
    Pivot,
    RsiRead,
    TimeframeRead,
    WaveRead,
)
from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories.research import ResearchRepository
from market_intel.exceptions import (
    ConfigurationError,
    MarketIntelError,
    ProviderError,
    RateLimitError,
)
from market_intel.providers.fundamentals import (
    Fundamentals,
    OptionQuote,
    OptionsSnapshot,
    YFinanceFundamentalsProvider,
)
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

#: Target delta for the LEAPS leg — deep enough to behave like stock.
LEAPS_TARGET_DELTA = 0.70
#: Minimum days to expiry for a contract to count as a LEAPS.
LEAPS_MIN_DAYS = 180
#: Target delta (absolute) for the cash-secured put.
CSP_TARGET_DELTA = 0.30
#: Days-to-expiry window for the wheel leg.
CSP_DTE_RANGE = (21, 60)
#: Starting position size, before the confidence and override adjustments.
BASE_POSITION_PCT = 25.0
#: Cash floor the rotation rules never breach.
CASH_FLOOR_PCT = 20.0


@dataclass(frozen=True, slots=True)
class OptionsRead:
    """Framework 9 — what the chain is saying."""

    put_call_oi: float | None = None
    put_call_volume: float | None = None
    max_pain: float | None = None
    max_pain_expiry: dt.date | None = None
    atm_iv: float | None = None
    realised_vol: float | None = None
    iv_premium: float | None = None  # ATM IV / realised vol
    realised_vol_percentile: float | None = None
    unusual: tuple[str, ...] = ()
    catalyst_expansion: str = ""
    available: bool = False
    note: str = ""

    def to_signal(self) -> FrameworkSignal:
        if not self.available:
            return FrameworkSignal("Options Flow", "n/a", "—", self.note or "no chain")
        parts = []
        if self.put_call_oi is not None:
            parts.append(f"P/C OI {self.put_call_oi:.2f}")
        if self.max_pain is not None:
            parts.append(f"max pain {self.max_pain:,.2f}")
        if self.atm_iv is not None:
            parts.append(f"ATM IV {self.atm_iv * 100:.0f}%")
        detail = " · ".join(parts) or "chain loaded"
        if self.put_call_oi is None:
            return FrameworkSignal("Options Flow", "neutral", "weak", detail)
        if self.put_call_oi < 0.7:
            return FrameworkSignal("Options Flow", "bullish", "moderate", detail)
        if self.put_call_oi > 1.3:
            return FrameworkSignal("Options Flow", "bearish", "moderate", detail)
        return FrameworkSignal("Options Flow", "neutral", "weak", detail)


@dataclass(frozen=True, slots=True)
class CatalystRead:
    """Framework 10 — earnings, estimates, targets, insiders, PEG."""

    fundamentals: Fundamentals | None = None
    days_to_earnings: int | None = None
    upside_to_mean: float | None = None
    peg_alert: str = ""  # "" | "PEG ALERT" | "HIGHEST PRIORITY"
    available: bool = False
    note: str = ""

    def to_signal(self) -> FrameworkSignal:
        if not self.available or self.fundamentals is None:
            return FrameworkSignal(
                "Catalyst & Fundamentals", "n/a", "—", self.note or "no fundamentals"
            )
        parts = []
        if self.upside_to_mean is not None:
            parts.append(f"{self.upside_to_mean:+.0f}% to analyst avg")
        if self.fundamentals.peg_ratio is not None:
            parts.append(f"PEG {self.fundamentals.peg_ratio:.2f}")
        if self.days_to_earnings is not None:
            parts.append(f"earnings in {self.days_to_earnings}d")
        detail = " · ".join(parts) or "limited coverage"
        if self.peg_alert == "HIGHEST PRIORITY":
            return FrameworkSignal(
                "Catalyst & Fundamentals", "bullish", "strong", detail + " · PEG < 0.5"
            )
        if self.upside_to_mean is None:
            return FrameworkSignal("Catalyst & Fundamentals", "neutral", "weak", detail)
        if self.upside_to_mean >= 25.0:
            return FrameworkSignal("Catalyst & Fundamentals", "bullish", "strong", detail)
        if self.upside_to_mean >= 5.0:
            return FrameworkSignal(
                "Catalyst & Fundamentals", "bullish", "moderate", detail
            )
        if self.upside_to_mean <= -5.0:
            return FrameworkSignal(
                "Catalyst & Fundamentals", "bearish", "moderate", detail
            )
        return FrameworkSignal("Catalyst & Fundamentals", "neutral", "weak", detail)


@dataclass(frozen=True, slots=True)
class UniLeg:
    """One of the five UNI legs."""

    name: str
    score: int
    basis: str
    computed: bool


@dataclass(frozen=True, slots=True)
class UniRead:
    """Framework 12 — the 5x5 quality screen."""

    legs: tuple[UniLeg, ...] = ()

    @property
    def total(self) -> int:
        return sum(leg.score for leg in self.legs)

    @property
    def judgement_legs(self) -> tuple[str, ...]:
        return tuple(leg.name for leg in self.legs if not leg.computed)

    def to_signal(self) -> FrameworkSignal:
        if not self.legs:
            return FrameworkSignal("UNI Score", "n/a", "—", "no fundamentals")
        detail = f"{self.total}/25"
        if self.judgement_legs:
            count = len(self.judgement_legs)
            plural = "legs need" if count != 1 else "leg needs"
            detail += f" ({count} {plural} judgement)"
        if self.total >= 19:
            return FrameworkSignal("UNI Score", "bullish", "strong", detail)
        if self.total >= 15:
            return FrameworkSignal("UNI Score", "bullish", "moderate", detail)
        if self.total <= 10:
            return FrameworkSignal("UNI Score", "bearish", "moderate", detail)
        return FrameworkSignal("UNI Score", "neutral", "weak", detail)


@dataclass(frozen=True, slots=True)
class LevelPlan:
    """The price map: invalidation, supports, targets."""

    current: float
    stop: float | None
    atr_stop: float | None
    supports: tuple[float, ...]
    target_1: float | None
    target_2: float | None
    extended: float | None
    target_basis: str = ""

    @property
    def risk_pct(self) -> float | None:
        if self.stop is None or not self.current:
            return None
        return (self.stop / self.current - 1.0) * 100.0

    @property
    def reward_pct(self) -> float | None:
        if self.target_1 is None or not self.current:
            return None
        return (self.target_1 / self.current - 1.0) * 100.0

    @property
    def reward_risk(self) -> float | None:
        if self.risk_pct is None or self.reward_pct is None or self.risk_pct == 0:
            return None
        return abs(self.reward_pct / self.risk_pct)


@dataclass(frozen=True, slots=True)
class RotationRead:
    """Framework 14 — sizing and the profit-taking ladder."""

    entry_size_pct: float
    first_tranche_pct: float
    scale_note: str
    reentry_zone: tuple[float, float] | None
    cash_floor_pct: float = CASH_FLOOR_PCT

    def to_signal(self) -> FrameworkSignal:
        detail = (
            f"{self.entry_size_pct:.0f}% of position now, "
            f"sell 50% at T1, rest at T2, {self.cash_floor_pct:.0f}% cash floor"
        )
        if self.entry_size_pct >= 20.0:
            return FrameworkSignal("Capital Rotation", "bullish", "moderate", detail)
        if self.entry_size_pct <= 5.0:
            return FrameworkSignal("Capital Rotation", "bearish", "moderate", detail)
        return FrameworkSignal("Capital Rotation", "neutral", "weak", detail)


@dataclass(frozen=True, slots=True)
class EntryPlan:
    """Whether to buy now, at what price, and what alerts to set."""

    decision: str  # "Y" | "N" | "Partial"
    trigger: float | None
    rationale: str
    alerts: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class LeapsPlan:
    """A concrete long-dated call, priced and stress-tested.

    Scenarios reprice the contract with Black-Scholes at the shorter remaining
    maturity rather than multiplying by delta, so time decay is included. That
    matters for a holder who exits inside two months: a linear delta estimate
    flatters a slow grind higher, which is exactly the case the ladder is
    meant to manage.
    """

    expiry: dt.date | None = None
    strike: float | None = None
    delta: float | None = None
    implied_vol: float | None = None
    premium: float | None = None
    breakeven: float | None = None
    scenarios: tuple[tuple[str, float, float, str], ...] = ()  # label, stock %, option %, action
    pmcc_eligible: bool = False
    pmcc_note: str = ""
    available: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class WheelPlan:
    """The cash-secured put leg and its covered-call follow-through."""

    expiry: dt.date | None = None
    strike: float | None = None
    delta: float | None = None
    premium: float | None = None
    probability_otm: float | None = None
    cost_basis: float | None = None
    covered_call_strike: float | None = None
    available: bool = False
    note: str = ""


@dataclass(frozen=True, slots=True)
class MkrAnalysis:
    """Everything the fourteen frameworks produced for one ticker."""

    symbol: str
    name: str | None
    generated_at: dt.datetime
    as_of: dt.date
    price: float
    previous_close: float | None
    week52_low: float | None
    week52_high: float | None
    currency: str | None

    wave: WaveRead
    fib: FibRead
    patterns: PatternRead
    rsi: RsiRead
    fvg: FvgRead
    flow: FlowRead
    moving_averages: MaRead
    momentum: MomentumRead
    options: OptionsRead
    catalyst: CatalystRead
    timeframes: TimeframeRead
    uni: UniRead
    confirmation: ConfirmationRead
    rotation: RotationRead

    exhaustion: ExhaustionRead
    levels: LevelPlan
    entry: EntryPlan
    leaps: LeapsPlan
    wheel: WheelPlan
    monte_carlo_thesis: MonteCarloResult | None
    monte_carlo_neutral: MonteCarloResult | None
    risks: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    pivots: tuple[Pivot, ...] = ()

    @property
    def market_signals(self) -> list[FrameworkSignal]:
        """The thirteen independent reads confidence is computed from.

        Capital rotation is excluded: it is sized *from* confidence, so feeding
        it back in would both double-count and make the number depend on itself.
        """
        return self.signals[:-1]

    @property
    def signals(self) -> list[FrameworkSignal]:
        """The fourteen scorecard rows, in framework order."""
        return [
            self.wave.to_signal(),
            self.fib.to_signal(),
            self.patterns.to_signal(),
            self.rsi.to_signal(),
            self.fvg.to_signal(),
            self.flow.to_signal(),
            self.moving_averages.to_signal(),
            self.momentum.to_signal(),
            self.options.to_signal(),
            self.catalyst.to_signal(),
            self.timeframes.to_signal(),
            self.uni.to_signal(),
            self.confirmation.to_signal(),
            self.rotation.to_signal(),
        ]

    @property
    def confidence(self) -> int:
        """0-100 directional conviction, from the thirteen independent reads."""
        return calc.confidence_score(
            self.market_signals, override=self.confirmation.death_cross_override
        )

    @property
    def change_pct(self) -> float | None:
        if not self.previous_close:
            return None
        return (self.price / self.previous_close - 1.0) * 100.0


class MkrService:
    """Builds, renders and optionally AI-writes the 14-framework analysis."""

    def __init__(
        self,
        db: Database,
        market_data: MarketDataService,
        settings: Settings,
        cache: ApiCache | None = None,
        fundamentals_provider: Any | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._db = db
        self._market_data = market_data
        self._settings = settings
        self._cache = cache
        self._fundamentals = fundamentals_provider or YFinanceFundamentalsProvider()
        self._client = client

    @property
    def ai_available(self) -> bool:
        """True when an Anthropic key is configured (or a client was injected)."""
        return self._client is not None or bool(self._settings.anthropic_api_key)

    # --- Stage 1 + 2: gather and analyse --------------------------------------

    def analyse(
        self,
        symbol: str,
        now: dt.datetime | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> MkrAnalysis:
        """Run all fourteen frameworks against ``symbol``.

        Args:
            symbol: Ticker to analyse (case-insensitive).
            now: Override the clock (tests).
            progress: Optional callback receiving a one-line status per stage.

        Raises:
            DataNotFoundError: No price history at all — nothing can be computed.
        """
        symbol = symbol.strip().upper()
        now = now or dt.datetime.now(dt.UTC)
        report = progress or (lambda _message: None)
        unavailable: list[str] = []

        report(f"Loading {symbol} price history...")
        end = now.date()
        start = end - dt.timedelta(days=self._settings.mkr_history_days)
        daily = self._market_data.get_price_history(symbol, start=start, end=end)
        if daily.empty:
            raise ProviderError(f"No price history for '{symbol}'", provider="market_data")

        close = daily["close"].astype(float)
        price = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if len(close) >= 2 else None
        as_of = pd.Timestamp(daily.index[-1]).date()

        weekly = calc.resample_ohlcv(daily, "W")
        monthly = calc.resample_ohlcv(daily, "ME")

        report("Fetching intraday bars...")
        intraday = self._safe(
            lambda: self._fundamentals.get_intraday(symbol),
            "4-hour fair value gaps",
            unavailable,
        )
        report("Fetching fundamentals...")
        fundamentals = self._cached_fundamentals(symbol, unavailable)
        report("Fetching option chain...")
        chain = self._cached_options(symbol, unavailable)

        report("Running the frameworks...")
        pivots = calc.find_pivots(daily)
        wave = calc.wave_read(pivots, price)
        fib = calc.fib_read(pivots, price)
        patterns = calc.pattern_read(daily, pivots)
        rsi = calc.rsi_read(daily, weekly)
        fvg = calc.fvg_read(daily, price, intraday)
        flow = calc.flow_read(daily)
        moving_averages = calc.ma_read(daily, pivots, price)
        momentum = calc.momentum_read(daily)
        timeframes = calc.timeframe_read(daily, weekly, monthly)
        confirmation = calc.confirmation_read(daily)
        exhaustion = calc.exhaustion_read(momentum, rsi, flow)

        options = self._options_read(chain, daily, price, fundamentals)
        catalyst = self._catalyst_read(fundamentals, price, end)
        uni = self._uni_read(fundamentals, price)

        window = close.iloc[-252:]
        week52_low = float(window.min()) if len(window) else None
        week52_high = float(window.max()) if len(window) else None
        if fundamentals is not None:
            week52_low = fundamentals.fifty_two_week_low or week52_low
            week52_high = fundamentals.fifty_two_week_high or week52_high

        levels = self._level_plan(
            price, fib, moving_averages, momentum, wave, fundamentals, week52_high
        )
        # Confidence is fixed here, from the thirteen independent reads, and the
        # sizing and entry rules are derived from it — so the scorecard, the
        # executive summary and the entry rationale all quote the same number.
        market_signals = [
            wave.to_signal(), fib.to_signal(), patterns.to_signal(), rsi.to_signal(),
            fvg.to_signal(), flow.to_signal(), moving_averages.to_signal(),
            momentum.to_signal(), options.to_signal(), catalyst.to_signal(),
            timeframes.to_signal(), uni.to_signal(), confirmation.to_signal(),
        ]
        confidence = calc.confidence_score(
            market_signals, override=confirmation.death_cross_override
        )
        currency = fundamentals.currency if fundamentals else None
        rotation = self._rotation_read(
            confidence, confirmation, exhaustion, levels, fib
        )
        entry = self._entry_plan(
            price, confidence, confirmation, exhaustion, levels,
            moving_averages, timeframes, currency,
        )

        report("Pricing the options legs...")
        leaps = self._leaps_plan(chain, price, momentum, options, unavailable)
        wheel = self._wheel_plan(chain, price, levels, unavailable)

        report("Running the Monte Carlo...")
        horizon = self._settings.mkr_horizon_days
        paths = self._settings.mkr_monte_carlo_paths
        neutral = calc.monte_carlo(close, "Neutral", horizon, paths)
        thesis_drift = None
        if levels.target_1 and price > 0:
            thesis_drift = math.log(levels.target_1 / price) / horizon * 252.0
        thesis = calc.monte_carlo(
            close, "Thesis-tilted", horizon, paths, annual_drift=thesis_drift
        )

        risks = self._risks(
            price, levels, confirmation, exhaustion, catalyst, options, fvg, neutral,
            currency,
        )

        return MkrAnalysis(
            symbol=symbol,
            name=(fundamentals.name if fundamentals else None),
            generated_at=now,
            as_of=as_of,
            price=price,
            previous_close=previous,
            week52_low=week52_low,
            week52_high=week52_high,
            currency=currency,
            wave=wave,
            fib=fib,
            patterns=patterns,
            rsi=rsi,
            fvg=fvg,
            flow=flow,
            moving_averages=moving_averages,
            momentum=momentum,
            options=options,
            catalyst=catalyst,
            timeframes=timeframes,
            uni=uni,
            confirmation=confirmation,
            rotation=rotation,
            exhaustion=exhaustion,
            levels=levels,
            entry=entry,
            leaps=leaps,
            wheel=wheel,
            monte_carlo_thesis=thesis,
            monte_carlo_neutral=neutral,
            risks=tuple(risks),
            unavailable=tuple(unavailable),
            pivots=tuple(pivots[-8:]),
        )

    # --- Framework builders that need more than price -------------------------

    def _options_read(
        self,
        chain: OptionsSnapshot | None,
        daily: pd.DataFrame,
        price: float,
        fundamentals: Fundamentals | None,
    ) -> OptionsRead:
        """Framework 9, with an honest substitute for IV rank.

        A real IV rank needs a year of implied-volatility history, which no
        free feed publishes. Quoting one anyway would be inventing a number, so
        ATM IV is compared against *realised* volatility and its one-year
        percentile instead, and the report says that is what it is.
        """
        realised = None
        percentile = None
        returns = np.diff(np.log(daily["close"].astype(float).to_numpy()))
        if returns.size >= 40:
            realised = float(np.std(returns[-30:], ddof=1) * math.sqrt(252))
            rolling = (
                pd.Series(returns).rolling(30).std(ddof=1) * math.sqrt(252)
            ).dropna()
            if len(rolling) >= 60:
                percentile = float((rolling.iloc[-252:] < realised).mean() * 100.0)

        if chain is None:
            return OptionsRead(
                realised_vol=realised,
                realised_vol_percentile=percentile,
                available=False,
                note="no listed options or chain unavailable",
            )

        premium = None
        if chain.atm_iv and realised:
            premium = chain.atm_iv / realised

        expansion = "no dated catalyst inside the front expiry"
        earnings = fundamentals.next_earnings if fundamentals else None
        if earnings and chain.max_pain_expiry and earnings <= chain.max_pain_expiry:
            expansion = (
                f"earnings {earnings:%d %b} lands before the front expiry — "
                "expect IV expansion into it and crush after"
            )
        elif earnings:
            expansion = f"earnings {earnings:%d %b} is beyond the front expiry"

        return OptionsRead(
            put_call_oi=chain.put_call_oi,
            put_call_volume=chain.put_call_volume,
            max_pain=chain.max_pain,
            max_pain_expiry=chain.max_pain_expiry,
            atm_iv=chain.atm_iv,
            realised_vol=realised,
            iv_premium=premium,
            realised_vol_percentile=percentile,
            unusual=chain.unusual,
            catalyst_expansion=expansion,
            available=True,
        )

    def _catalyst_read(
        self, fundamentals: Fundamentals | None, price: float, today: dt.date
    ) -> CatalystRead:
        """Framework 10, including the PEG alert ladder."""
        if fundamentals is None:
            return CatalystRead(note="fundamentals unavailable")
        days = None
        if fundamentals.next_earnings:
            days = (fundamentals.next_earnings - today).days
        upside = None
        if fundamentals.target_mean and price:
            upside = (fundamentals.target_mean / price - 1.0) * 100.0
        alert = ""
        if fundamentals.peg_ratio is not None and fundamentals.peg_ratio > 0:
            if fundamentals.peg_ratio < 0.5:
                alert = "HIGHEST PRIORITY"
            elif fundamentals.peg_ratio < 1.0:
                alert = "PEG ALERT"
        return CatalystRead(fundamentals, days, upside, alert, available=True)

    def _uni_read(self, fundamentals: Fundamentals | None, price: float) -> UniRead:
        """Framework 12 — three legs computable, two need judgement.

        Scarcity/monopoly IP and pricing power cannot be read off a price feed.
        They are scored a neutral 3 and listed in ``judgement_legs`` so the
        report says which part of the total is a placeholder — a 21/25 built on
        two guesses should not read the same as a 21/25 that was measured.
        """
        if fundamentals is None:
            return UniRead()

        legs: list[UniLeg] = [
            UniLeg(
                "Physical scarcity / monopoly IP", 3,
                "not derivable from market data — judgement layer to score", False,
            ),
        ]

        margin = fundamentals.profit_margin
        if margin is None:
            legs.append(
                UniLeg("Pricing power", 3, "no margin data — judgement layer", False)
            )
        else:
            score = _band(margin, (0.30, 0.20, 0.10, 0.0))
            legs.append(
                UniLeg("Pricing power", score, f"net margin {margin * 100:.1f}%", True)
            )

        cap = fundamentals.market_cap
        if cap is None:
            legs.append(UniLeg("Market cap < $20B", 3, "market cap unknown", False))
        else:
            billions = cap / 1e9
            # Inverted: this leg rewards being small, so the bands run the
            # other way and 100bn+ scores the floor.
            score = 5 if billions < 5 else 4 if billions < 20 else 2 if billions < 100 else 1
            # The $20bn threshold is the framework's own; the value carries the
            # listing currency, which is not the dollar for every name.
            unit = f"B {fundamentals.currency}" if fundamentals.currency else "B"
            legs.append(
                UniLeg("Market cap < $20B", score, f"{billions:,.1f}{unit}", True)
            )

        growth = fundamentals.revenue_growth
        if growth is None:
            legs.append(UniLeg("Revenue inflection", 3, "no revenue growth data", False))
        else:
            score = _band(growth, (0.40, 0.20, 0.05, 0.0))
            legs.append(
                UniLeg("Revenue inflection", score, f"revenue {growth * 100:+.1f}% y/y", True)
            )

        if fundamentals.target_mean and price:
            upside = (fundamentals.target_mean / price - 1.0) * 100.0
            score = _band(upside, (50.0, 30.0, 15.0, 0.0))
            legs.append(
                UniLeg(
                    ">50% upside to analyst avg", score,
                    f"{upside:+.1f}% to "
                    + money_formatter(price, fundamentals.currency)(
                        fundamentals.target_mean
                    ),
                    True,
                )
            )
        else:
            legs.append(
                UniLeg(">50% upside to analyst avg", 3, "no analyst target", False)
            )

        return UniRead(tuple(legs))

    # --- Plans ----------------------------------------------------------------

    def _level_plan(
        self,
        price: float,
        fib: FibRead,
        moving_averages: MaRead,
        momentum: MomentumRead,
        wave: WaveRead,
        fundamentals: Fundamentals | None,
        week52_high: float | None,
    ) -> LevelPlan:
        """Invalidation, supports and the three upside targets.

        The stop is structural — the nearest confirmed swing low beneath spot,
        or the wave invalidation when that is tighter. The ATR stop is reported
        alongside it rather than replacing it: when structure sits further away
        than two ATRs, the right response is a smaller position, not a tighter
        stop placed where noise will hit it.
        """
        # The invalidation must be structural — a confirmed swing low, or the
        # wave invalidation when that is nearer. A moving average is a level to
        # watch, not the place a thesis breaks, so the averages are shown as
        # support but never chosen as the stop.
        structural = moving_averages.supports[0] if moving_averages.supports else None
        if wave.invalidation is not None and wave.invalidation < price:
            structural = (
                max(structural, wave.invalidation) if structural else wave.invalidation
            )
        atr_stop = (
            price - ATR_STOP_MULTIPLE * momentum.atr if momentum.atr else None
        )
        stop = structural if structural is not None else atr_stop

        displayed = [
            level
            for level in (
                moving_averages.ema20, moving_averages.ema50,
                moving_averages.vwap20, *moving_averages.supports, stop,
            )
            if level is not None and level < price
        ]
        supports = sorted({round(level, 4) for level in displayed}, reverse=True)

        # A target inside the daily noise is not a target. Anything closer than
        # one ATR (or 1.5% on a quiet name) is skipped — that is what stops a
        # stock sitting just under its own prior high from reporting a 0.1x
        # reward-to-risk as though it were a trade.
        floor = max(momentum.atr or 0.0, price * 0.015)
        candidates: list[tuple[float, str]] = []
        for level in moving_averages.resistances:
            candidates.append((level, "prior swing high"))
        for label, level in fib.extensions.items():
            if level > price:
                candidates.append((level, f"fib {label} extension"))
        if week52_high and week52_high > price:
            candidates.append((week52_high, "52-week high"))
        if moving_averages.all_time_high and moving_averages.all_time_high > price:
            candidates.append((moving_averages.all_time_high, "all-time high"))
        if fundamentals and fundamentals.target_mean and fundamentals.target_mean > price:
            candidates.append((fundamentals.target_mean, "analyst mean target"))
        candidates = [item for item in candidates if item[0] >= price + floor]
        candidates.sort(key=lambda item: item[0])

        target_1 = candidates[0][0] if candidates else None
        target_2 = candidates[1][0] if len(candidates) > 1 else None
        extended = fib.extensions.get("2.618x")
        if extended is None or extended <= price:
            extended = candidates[-1][0] if candidates else None
        basis = " · ".join(
            f"T{index + 1} {label}" for index, (_, label) in enumerate(candidates[:2])
        )
        return LevelPlan(
            price, stop, atr_stop, tuple(supports[:4]), target_1, target_2, extended, basis
        )

    def _entry_plan(
        self,
        price: float,
        confidence: int,
        confirmation: ConfirmationRead,
        exhaustion: ExhaustionRead,
        levels: LevelPlan,
        moving_averages: MaRead,
        timeframes: TimeframeRead,
        currency: str | None = None,
    ) -> EntryPlan:
        """Buy now, wait, or scale in — and at exactly what price."""
        money_static = money_formatter(price, currency)
        alerts: list[tuple[str, float]] = []
        if levels.stop is not None:
            alerts.append(("stop", levels.stop))

        if confirmation.death_cross_override:
            trigger = moving_averages.ema50
            if trigger:
                alerts.append(("reclaim", trigger))
            return EntryPlan(
                "N", trigger,
                "Death cross override active: 50 EMA below 200 EMA with price beneath "
                "the 50. No long entry until price reclaims the 50 EMA.",
                tuple(alerts),
            )

        if exhaustion.triggered:
            pullback = levels.supports[0] if levels.supports else None
            if pullback:
                alerts.append(("pullback entry", pullback))
            return EntryPlan(
                "N", pullback,
                f"Exhaustion override {exhaustion.summary}. Wait for the reset rather "
                "than paying up into it.",
                tuple(alerts),
            )

        breakout = moving_averages.resistances[0] if moving_averages.resistances else None
        reward_risk = levels.reward_risk
        if confidence >= 65 and timeframes.aligned:
            # Conviction is not the same as a payout. Where the first target is
            # closer than the invalidation, a full-size entry at market is a bad
            # trade however green the board is — take a starter instead and let
            # the pullback improve the entry.
            if reward_risk is not None and reward_risk < 1.0:
                alerts.append(("entry", price))
                if levels.supports:
                    alerts.append(("add on pullback", levels.supports[0]))
                return EntryPlan(
                    "Partial", price,
                    f"Confidence {confidence}/100 with every timeframe aligned, but T1 "
                    f"({money_static(levels.target_1)}) sits closer than the stop "
                    f"({money_static(levels.stop)}) — only {reward_risk:.2f}x reward to "
                    "risk. Starter size here; add on a pullback to support.",
                    tuple(alerts),
                )
            alerts.append(("entry", price))
            return EntryPlan(
                "Y", price,
                f"Confidence {confidence}/100 with all timeframes aligned and "
                + (
                    f"{reward_risk:.1f}x reward to risk"
                    if reward_risk is not None
                    else "no defined target yet"
                )
                + " — the setup is already working; enter at market and manage "
                "against the stop.",
                tuple(alerts),
            )
        if confidence >= 50:
            trigger = breakout or price
            alerts.append(("entry", trigger))
            return EntryPlan(
                "Partial", trigger,
                f"Confidence {confidence}/100 but timeframes are mixed. Take a starter "
                f"here and add on a close above {money_static(trigger)}.",
                tuple(alerts),
            )
        reclaim = moving_averages.ema20 if moving_averages.ema20 else breakout
        if reclaim:
            alerts.append(("reclaim", reclaim))
        return EntryPlan(
            "N", reclaim,
            f"Confidence {confidence}/100. The board is not paying for risk here — "
            "wait for momentum to turn.",
            tuple(alerts),
        )

    def _rotation_read(
        self,
        confidence: int,
        confirmation: ConfirmationRead,
        exhaustion: ExhaustionRead,
        levels: LevelPlan,
        fib: FibRead,
    ) -> RotationRead:
        """Framework 14 — size from conviction, never below the cash floor."""
        size = BASE_POSITION_PCT * (confidence / 100.0) * 2.0
        if confirmation.death_cross_override:
            size = 0.0
        elif exhaustion.triggered:
            size = min(size, 10.0)
        size = max(0.0, min(size, 100.0 - CASH_FLOOR_PCT))

        reentry = None
        if fib.golden_pocket:
            reentry = fib.golden_pocket
        elif levels.supports:
            floor = levels.supports[0]
            reentry = (floor * 0.98, floor * 1.02)

        return RotationRead(
            entry_size_pct=size,
            first_tranche_pct=50.0,
            scale_note=(
                "Sell 50% into T1 (the W3 peak), hold the rest for T2 (W5), "
                "trail the remainder against the 20 EMA."
            ),
            reentry_zone=reentry,
        )

    def _leaps_plan(
        self,
        chain: OptionsSnapshot | None,
        price: float,
        momentum: MomentumRead,
        options: OptionsRead,
        unavailable: list[str],
    ) -> LeapsPlan:
        """Pick a ~0.70-delta long-dated call and reprice it under four paths."""
        if chain is None:
            return LeapsPlan(note="no option chain available")
        del unavailable  # the chain failure is already recorded by the caller
        today = dt.date.today()
        long_dated = [
            expiry
            for expiry in chain.expiries
            if (expiry - today).days >= LEAPS_MIN_DAYS
        ]
        if not long_dated:
            return LeapsPlan(
                note=f"no expiry {LEAPS_MIN_DAYS}+ days out in the loaded chain"
            )
        # Nearest to a year out rather than the furthest listed: a two-year
        # contract ties up far more premium for the same delta, and this plan is
        # held for weeks.
        expiry = min(long_dated, key=lambda value: abs((value - today).days - 365))
        years = max((expiry - today).days / 365.0, 1e-6)
        rate = self._settings.mkr_risk_free_rate

        calls = [
            quote
            for quote in chain.for_expiry(expiry, "call")
            if quote.implied_vol and quote.implied_vol > 0 and quote.mid
        ]
        if not calls:
            return LeapsPlan(note=f"no priced calls for {expiry:%b %Y}")

        scored: list[tuple[float, OptionQuote, float]] = []
        for quote in calls:
            delta = calc.bs_delta(price, quote.strike, years, quote.implied_vol, rate)
            if delta is None:
                continue
            scored.append((abs(delta - LEAPS_TARGET_DELTA), quote, delta))
        if not scored:
            return LeapsPlan(note="could not compute delta for any listed call")
        scored.sort(key=lambda item: item[0])
        _, contract, delta = scored[0]
        premium = contract.mid or 0.0
        vol = contract.implied_vol or 0.0
        breakeven = contract.strike + premium

        # Expected moves come from the security's own implied vol, not a guess:
        # a one-sigma 30-day move is vol * sqrt(30/365).
        sigma_30 = vol * math.sqrt(30.0 / 365.0)
        sigma_60 = vol * math.sqrt(60.0 / 365.0)
        scenarios: list[tuple[str, float, float, str]] = []
        for label, days, move, action in (
            ("30d slow", 30, 0.5 * sigma_30, "SELL HALF at +50%"),
            ("30d fast", 30, 1.5 * sigma_30, "HOLD, let it run"),
            ("60d base", 60, 0.5 * sigma_60, "trim into strength"),
            ("60d rip", 60, 1.5 * sigma_60, "FULL HOLD"),
        ):
            future_spot = price * (1.0 + move)
            remaining = max(years - days / 365.0, 1e-6)
            value = calc.bs_price(future_spot, contract.strike, remaining, vol, rate)
            if value is None or premium <= 0:
                continue
            scenarios.append(
                (label, move * 100.0, (value / premium - 1.0) * 100.0, action)
            )

        pmcc = delta >= 0.75
        note = (
            "LEAPS delta is deep enough to cover a short call"
            if pmcc
            else f"delta {delta:.2f} is below the 0.75 a PMCC needs"
        )
        return LeapsPlan(
            expiry=expiry,
            strike=contract.strike,
            delta=delta,
            implied_vol=vol,
            premium=premium,
            breakeven=breakeven,
            scenarios=tuple(scenarios),
            pmcc_eligible=pmcc,
            pmcc_note=note,
            available=True,
        )

    def _wheel_plan(
        self,
        chain: OptionsSnapshot | None,
        price: float,
        levels: LevelPlan,
        unavailable: list[str],
    ) -> WheelPlan:
        """Pick the ~30-delta put in the 21-60 day window."""
        if chain is None:
            return WheelPlan(note="no option chain available")
        del unavailable  # the chain failure is already recorded by the caller
        today = dt.date.today()
        window = [
            expiry
            for expiry in chain.expiries
            if CSP_DTE_RANGE[0] <= (expiry - today).days <= CSP_DTE_RANGE[1]
        ]
        if not window:
            return WheelPlan(
                note=f"no expiry {CSP_DTE_RANGE[0]}-{CSP_DTE_RANGE[1]} days out"
            )
        expiry = min(window)
        years = max((expiry - today).days / 365.0, 1e-6)
        rate = self._settings.mkr_risk_free_rate

        best: tuple[float, OptionQuote, float] | None = None
        for quote in chain.for_expiry(expiry, "put"):
            if not quote.implied_vol or quote.implied_vol <= 0 or not quote.mid:
                continue
            delta = calc.bs_delta(
                price, quote.strike, years, quote.implied_vol, rate, kind="put"
            )
            if delta is None:
                continue
            distance = abs(abs(delta) - CSP_TARGET_DELTA)
            if best is None or distance < best[0]:
                best = (distance, quote, delta)
        if best is None:
            return WheelPlan(note=f"no priced puts for {expiry:%d %b %Y}")

        _, contract, delta = best
        premium = contract.mid or 0.0
        return WheelPlan(
            expiry=expiry,
            strike=contract.strike,
            delta=delta,
            premium=premium,
            probability_otm=(1.0 - abs(delta)) * 100.0,
            cost_basis=contract.strike - premium,
            covered_call_strike=levels.target_1,
            available=True,
        )

    def _risks(
        self,
        price: float,
        levels: LevelPlan,
        confirmation: ConfirmationRead,
        exhaustion: ExhaustionRead,
        catalyst: CatalystRead,
        options: OptionsRead,
        fvg: FvgRead,
        neutral: MonteCarloResult | None,
        currency: str | None = None,
    ) -> list[str]:
        """The three biggest risks, each pinned to a price or a date."""
        money = money_formatter(price, currency)
        risks: list[tuple[int, str]] = []
        if confirmation.death_cross_override and confirmation.ema50:
            risks.append((
                0,
                f"Death cross in force — every bullish count is void while price is "
                f"below the 50 EMA at {money(confirmation.ema50)}.",
            ))
        if levels.stop is not None:
            risks.append((
                1,
                f"Structural invalidation at {money(levels.stop)} "
                f"({(levels.stop / price - 1.0) * 100:+.1f}%): a daily close beneath it "
                "breaks the swing sequence the targets are built on.",
            ))
        if exhaustion.flags:
            risks.append((2, f"Exhaustion — {exhaustion.summary}."))
        if catalyst.days_to_earnings is not None and catalyst.days_to_earnings <= 45:
            earnings = catalyst.fundamentals.next_earnings if catalyst.fundamentals else None
            risks.append((
                1,
                f"Earnings on {earnings:%d %b} ({catalyst.days_to_earnings} days) is a "
                "binary inside the holding period; the option legs carry crush risk.",
            ))
        if fvg.nearest_below is not None:
            gap = fvg.nearest_below
            risks.append((
                3,
                f"Unfilled {gap.timeframe} gap at {money(gap.low)}-{money(gap.high)} "
                f"({(gap.midpoint / price - 1.0) * 100:+.1f}%) is an open magnet below.",
            ))
        if options.iv_premium is not None and options.iv_premium > 1.3:
            risks.append((
                3,
                f"Implied vol is {options.iv_premium:.1f}x realised — long premium is "
                "paying up for movement the stock has not delivered.",
            ))
        if neutral is not None:
            risks.append((
                4,
                f"Neutral Monte Carlo puts the worst 1% of 90-day paths at "
                f"{neutral.worst_1pct:.1f}% ({money(price * (1 + neutral.worst_1pct / 100))}).",
            ))
        risks.sort(key=lambda item: item[0])
        return [text for _, text in risks[:3]]

    # --- Stage 3: write -------------------------------------------------------

    def generate(
        self,
        analysis: MkrAnalysis,
        use_web_search: bool = True,
        extra_context: str | None = None,
        archive: bool = True,
    ) -> dict[str, Any]:
        """Have Claude write the report from the computed pack, and archive it.

        Args:
            analysis: The finished :class:`MkrAnalysis`.
            use_web_search: Let the model research catalysts and news online.
            extra_context: Anything the user wants the write-up to weigh.
            archive: Store the result as a research note.

        Returns:
            The note as a dict, plus a ``searches`` count.

        Raises:
            ConfigurationError: No API key configured.
            RateLimitError: The API rate limit was hit.
            ProviderError: Any other API failure, refusal, or empty response.
        """
        client = self._get_client()
        payload = format_analysis(analysis)
        instruction = (
            f"Write the MKR 14-Framework Analysis for {analysis.symbol}.\n\n"
            "Every computed number is in the pack below. Reproduce the OUTPUT FORMAT "
            "exactly. Your judgement is needed on four things and nothing else:\n"
            "1. The Elliott Wave count — the pack gives you the pivot sequence and a "
            "heuristic label; confirm, refine or overrule it, and say if it is ambiguous.\n"
            "2. Chart patterns the arithmetic cannot see: Cup & Handle and Head & "
            "Shoulders.\n"
            "3. The two UNI legs marked as needing judgement — score them 1-5 with a "
            "reason, then restate the total.\n"
            "4. The catalyst picture and the three risks: research what is actually "
            "scheduled and what the market is arguing about."
        )
        if use_web_search:
            instruction += (
                "\n\nUse the web search tool to check the catalyst calendar, recent "
                "news and anything that would change the read. Attribute what you find."
            )
        if extra_context:
            instruction += f"\n\nThe trader adds: {extra_context}"

        tools: list[dict[str, Any]] = []
        if use_web_search:
            tools.append(
                {
                    "type": "web_search_20260318",
                    "name": "web_search",
                    "max_uses": self._settings.mkr_web_search_max_uses,
                }
            )

        try:
            response = client.messages.create(
                model=self._settings.ai_model,
                max_tokens=self._settings.ai_max_tokens,
                system=_SYSTEM_PROMPT,
                tools=tools or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": f"{instruction}\n\n{payload}"}],
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitError(
                f"Anthropic rate limit hit: {exc}", provider="anthropic"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error ({exc.status_code}): {exc.message}",
                provider="anthropic",
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Could not reach the Anthropic API: {exc}", provider="anthropic"
            ) from exc

        if response.stop_reason == "refusal":
            raise ProviderError(
                "The model declined to write this analysis", provider="anthropic"
            )
        if response.stop_reason == "max_tokens":
            logger.warning("MKR analysis hit the max_tokens limit and may be truncated")

        content = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not content:
            raise ProviderError("Model returned an empty analysis", provider="anthropic")

        searches = sum(
            1 for block in response.content if block.type == "server_tool_use"
        )
        title = f"MKR 14-Framework - {analysis.symbol} - {analysis.as_of:%Y-%m-%d}"
        result: dict[str, Any] = {
            "title": title,
            "content": content,
            "model": response.model,
            "searches": searches,
        }
        if archive:
            with self._db.session() as session:
                note = ResearchRepository(session).add(
                    title=title,
                    content=content,
                    note_type="mkr_analysis",
                    model=response.model,
                    tags=f"mkr,{analysis.symbol}",
                )
                result.update(
                    id=note.id, created_at=note.created_at, tags=note.tags
                )
        logger.info("MKR analysis written for %s (%d searches)", analysis.symbol, searches)
        return result

    def archive_deterministic(self, analysis: MkrAnalysis) -> dict[str, Any]:
        """Store the computed report as a research note, with no model involved."""
        title = f"MKR 14-Framework - {analysis.symbol} - {analysis.as_of:%Y-%m-%d}"
        content = render_report(analysis)
        with self._db.session() as session:
            note = ResearchRepository(session).add(
                title=title,
                content=content,
                note_type="mkr_analysis",
                model=None,
                tags=f"mkr,{analysis.symbol}",
            )
            return {
                "id": note.id,
                "title": note.title,
                "content": note.content,
                "note_type": note.note_type,
                "model": None,
                "tags": note.tags,
                "created_at": note.created_at,
                "searches": 0,
            }

    # --- internals ------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise ConfigurationError(
                    "MIP_ANTHROPIC_API_KEY is not set - AI write-up is disabled"
                )
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client

    def _safe(
        self, call: Callable[[], Any], label: str, unavailable: list[str]
    ) -> Any | None:
        """Run an optional enrichment, recording failure instead of raising."""
        try:
            return call()
        except MarketIntelError as exc:
            logger.info("%s unavailable: %s", label, exc)
            unavailable.append(f"{label} ({exc})")
        except Exception as exc:  # noqa: BLE001 - an optional feed must never break the run
            logger.warning("%s failed unexpectedly: %s", label, exc)
            unavailable.append(f"{label} (unexpected error)")
        return None

    def _cached_fundamentals(
        self, symbol: str, unavailable: list[str]
    ) -> Fundamentals | None:
        key = f"mkr_fundamentals:{symbol}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return Fundamentals.from_payload(cached)
        result = self._safe(
            lambda: self._fundamentals.get_fundamentals(symbol),
            "fundamentals",
            unavailable,
        )
        if result is not None and self._cache is not None:
            self._cache.set(
                key, result.to_payload(), self._settings.fundamentals_cache_ttl_minutes
            )
        return result

    def _cached_options(
        self, symbol: str, unavailable: list[str]
    ) -> OptionsSnapshot | None:
        key = f"mkr_options:{symbol}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return OptionsSnapshot.from_payload(cached)
        result = self._safe(
            lambda: self._fundamentals.get_options(symbol), "option chain", unavailable
        )
        if result is not None and self._cache is not None:
            self._cache.set(
                key, result.to_payload(), self._settings.options_cache_ttl_minutes
            )
        return result


# --- Rendering ----------------------------------------------------------------

_SYSTEM_PROMPT = """You are an elite quantitative trading analyst running the \
MKR 14-Framework Analysis for a professional trader.

ABSOLUTE RULE ON NUMBERS: every price, level, ratio, percentage and date you \
write must come from the analysis pack you are given, or from a source you \
found with the web search tool and attribute inline. You may not estimate, \
round-trip or invent a number. If the pack marks something unavailable, write \
"not available" and say why — never fill the gap with a plausible figure.

The arithmetic has already been done. Do not recompute indicators, retracements \
or option greeks; quote what the pack states. Spend your effort on the four \
judgement calls the pack flags, on the catalyst research, and on writing a wrap \
a trader can act on in sixty seconds.

Reproduce the requested output format exactly, including the section headings, \
the fourteen scorecard rows and the emoji level markers. Be direct: if the \
setup is bad, say so in the first sentence."""


def render_report(analysis: MkrAnalysis) -> str:
    """The full report in the MKR output format, computed only.

    This is what the page shows without an API key, and the skeleton the model
    is asked to reproduce with one.
    """
    a = analysis
    money = money_formatter(a.price, a.currency)
    lines: list[str] = []

    lines.append(f"# MKR 14-Framework Analysis — {a.symbol}")
    if a.name:
        lines.append(f"*{a.name}*")
    week52 = "n/a"
    if a.week52_low and a.week52_high:
        week52 = f"{money(a.week52_low)} → {money(a.week52_high)}"
    change = f" ({a.change_pct:+.2f}%)" if a.change_pct is not None else ""
    lines.append(
        f"**TICKER:** {a.symbol} | **LIVE PRICE:** {money(a.price)}{change} | "
        f"**52WK:** {week52}"
    )
    lines.append(
        f"*Close of {a.as_of:%d %b %Y} · generated {a.generated_at:%Y-%m-%d %H:%M UTC}*"
    )
    lines.append("")

    # Executive summary
    lines.append("**EXECUTIVE SUMMARY**")
    lines.append("")
    lines.append(_executive_summary(a, money))
    lines.append("")

    # Scorecard
    lines.append("**SCORECARD TABLE**")
    lines.append("")
    lines.append("| Framework | Signal | Strength | Detail |")
    lines.append("| --- | --- | --- | --- |")
    for signal in a.signals:
        icon = _SIGNAL_ICON[signal.signal]
        lines.append(
            f"| {signal.name} | {icon} {signal.signal} | {signal.strength} | {signal.detail} |"
        )
    lines.append("")

    # Key levels
    lines.append("**KEY PRICE LEVELS**")
    lines.append("")
    lines.append(f"🔴 Stop: {money(a.levels.stop)} (invalidation)")
    supports = " / ".join(money(level) for level in a.levels.supports) or "none below"
    lines.append(f"🟡 Support: {supports}")
    lines.append(f"🟢 Current: {money(a.price)}")
    lines.append(f"🎯 Target 1: {money(a.levels.target_1)} (W3/T1)")
    lines.append(f"🎯 Target 2: {money(a.levels.target_2)} (W5/T2)")
    lines.append(f"🚀 Extended: {money(a.levels.extended)} (2.618x extension)")
    if a.levels.reward_risk:
        lines.append(
            f"*Reward/risk to T1: {a.levels.reward_risk:.2f}x "
            f"({a.levels.reward_pct:+.1f}% vs {a.levels.risk_pct:+.1f}%)*"
        )
    if a.levels.atr_stop:
        lines.append(f"*2x ATR stop for sizing: {money(a.levels.atr_stop)}*")
    lines.append("")

    # Framework detail
    lines.append("**FRAMEWORK DETAIL**")
    lines.append("")
    lines.extend(_framework_detail(a, money))
    lines.append("")

    # Entry timing
    lines.append("**ENTRY TIMING**")
    lines.append("")
    lines.append(f"Enter now: {a.entry.decision}")
    lines.append(f"Exact trigger price: {money(a.entry.trigger)}")
    alerts = " · ".join(f"{money(level)} ({label})" for label, level in a.entry.alerts)
    lines.append(f"Alerts to set: {alerts or 'none'}")
    lines.append("")
    lines.append(a.entry.rationale)
    lines.append("")

    # LEAPS
    lines.append("**LEAPS RECOMMENDATION**")
    lines.append("")
    if a.leaps.available:
        lines.append(
            f"Contract: {a.symbol} {a.leaps.expiry:%b %Y} {a.leaps.strike:g}C"
        )
        lines.append(
            f"IV: {a.leaps.implied_vol * 100:.0f}% · Delta: {a.leaps.delta:.2f} · "
            f"Premium: {money(a.leaps.premium)} · Break-even: {money(a.leaps.breakeven)}"
        )
        lines.append("")
        lines.append(
            f"*Holding period is capped at {_MAX_HOLD_MONTHS} months — these scenarios "
            "reprice the contract at the shorter maturity, so they show leverage net of "
            "decay, not value at expiry.*"
        )
        lines.append("")
        for horizon in ("30d", "60d"):
            rows = [row for row in a.leaps.scenarios if row[0].startswith(horizon)]
            if not rows:
                continue
            lines.append(f"{horizon} scenarios (1σ move = {_sigma_text(a.leaps, horizon)}):")
            for label, stock_move, option_move, action in rows:
                name = label.split(" ", 1)[1]
                lines.append(
                    f"- {name.title()} ({stock_move:+.1f}% stock → "
                    f"{option_move:+.0f}% option) → {action}"
                )
            lines.append("")
        lines.append(
            f"PMCC eligible: {'Y' if a.leaps.pmcc_eligible else 'N'} — {a.leaps.pmcc_note}"
        )
    else:
        lines.append(f"Not available — {a.leaps.note}.")
    lines.append("")

    # Wheel
    lines.append("**WHEEL / CSP**")
    lines.append("")
    if a.wheel.available:
        lines.append(
            f"Strike: {money(a.wheel.strike)} ({abs(a.wheel.delta):.2f} delta, "
            f"{a.wheel.expiry:%d %b %Y}) · Premium: {money(a.wheel.premium)} · "
            f"POP: ~{a.wheel.probability_otm:.0f}% · If assigned: cost basis "
            f"{money(a.wheel.cost_basis)}"
        )
        if a.wheel.covered_call_strike:
            lines.append(
                f"Covered call plan: sell the {money(a.wheel.covered_call_strike)} "
                "strike at T1."
            )
    else:
        lines.append(f"Not available — {a.wheel.note}.")
    lines.append("")

    # Verdict line
    peg = "n/a"
    if a.catalyst.fundamentals and a.catalyst.fundamentals.peg_ratio is not None:
        peg = f"{a.catalyst.fundamentals.peg_ratio:.2f}x"
        if a.catalyst.peg_alert:
            peg += f" [{a.catalyst.peg_alert}]"
    avoid = _avoid_condition(a, money)
    lines.append(
        f"**{a.symbol}: {a.uni.total}/25 · CONFIDENCE: {a.confidence}/100 · "
        f"AVOID IF: {avoid} · PEG: {peg}**"
    )
    lines.append("")

    lines.append("Top 3 Risks:")
    lines.append("")
    for index, risk in enumerate(a.risks, start=1):
        lines.append(f"{index}. {risk}")
    if not a.risks:
        lines.append("1. No structural risk flags fired — which is itself worth a check.")
    lines.append("")

    # Monte Carlo
    lines.append(
        f"**MONTE CARLO ({_paths(a):,} paths · {_horizon(a)} days)**"
    )
    lines.append("")
    for result in (a.monte_carlo_thesis, a.monte_carlo_neutral):
        if result is None:
            continue
        lines.append(
            f"{result.label}: Win rate {result.win_rate:.0f}% · "
            f"Avg win {result.average_win:+.1f}% · Avg loss {result.average_loss:+.1f}% · "
            f"Worst 1%: {result.worst_1pct:.1f}% · Median {result.median_return:+.1f}%"
        )
    if a.monte_carlo_neutral:
        lines.append("")
        lines.append(
            f"*Bootstrapped from the last year of this security's own daily returns "
            f"(annualised vol {a.monte_carlo_neutral.annual_vol:.0f}%). The thesis-tilted "
            f"run recentres the drift on the path to T1; volatility is unchanged.*"
        )
        if (
            a.monte_carlo_thesis
            and a.monte_carlo_thesis.annual_drift < a.monte_carlo_neutral.annual_drift
        ):
            lines.append("")
            lines.append(
                f"*The tilted run is the more conservative one here: reaching T1 implies "
                f"{a.monte_carlo_thesis.annual_drift:.0f}% annualised, below the "
                f"{a.monte_carlo_neutral.annual_drift:.0f}% this security has actually "
                "delivered over the past year. T1 is a near target, not a stretch.*"
            )
    lines.append("")

    if a.unavailable:
        lines.append("**DATA NOT AVAILABLE**")
        lines.append("")
        for item in a.unavailable:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip()


def format_analysis(analysis: MkrAnalysis) -> str:
    """The fact pack handed to the model.

    It is the deterministic report plus the raw structure the judgement layer
    needs — the pivot sequence, the full retracement grid, the unfilled gaps
    and the UNI legs still awaiting a score. Everything the model is permitted
    to quote appears here, which is what makes the never-invent-a-number rule
    enforceable.
    """
    a = analysis
    money = money_formatter(a.price, a.currency)
    lines = [render_report(a), "", "---", "", "## Raw structure for the judgement layer", ""]

    lines.append("### Pivot sequence (oldest first)")
    for pivot in a.pivots:
        lines.append(f"- {pivot.date:%Y-%m-%d} {pivot.kind} {money(pivot.price)}")
    if not a.pivots:
        lines.append("- none confirmed")
    lines.append("")
    lines.append(
        f"Heuristic wave label: {a.wave.position} — {a.wave.reason} "
        f"(ambiguous: {'yes' if a.wave.ambiguous else 'no'})"
    )
    lines.append("")

    if a.fib.swing:
        swing = a.fib.swing
        lines.append(
            f"### Fibonacci grid ({'up' if swing.is_up else 'down'} leg "
            f"{money(swing.start.price)} {swing.start.date:%d %b} → "
            f"{money(swing.end.price)} {swing.end.date:%d %b})"
        )
        for label, level in a.fib.retracements.items():
            lines.append(f"- {label}: {money(level)}")
        if a.fib.golden_pocket:
            lines.append(
                f"- Golden pocket: {money(a.fib.golden_pocket[0])}-{money(a.fib.golden_pocket[1])}"
            )
        for label, level in a.fib.extensions.items():
            lines.append(f"- Extension {label}: {money(level)} (from {a.fib.extension_anchor})")
        lines.append("")

    if a.fvg.unfilled:
        lines.append("### Unfilled fair value gaps")
        for gap in a.fvg.unfilled[-8:]:
            lines.append(
                f"- {gap.timeframe} {gap.direction} {gap.date:%Y-%m-%d}: "
                f"{money(gap.low)}-{money(gap.high)}"
            )
        lines.append("")

    if a.patterns.model_checks:
        lines.append("### Patterns the arithmetic cannot see — your call")
        for name in a.patterns.model_checks:
            lines.append(f"- {name}: not evaluated")
        lines.append("")

    lines.append("### UNI legs")
    for leg in a.uni.legs:
        marker = "computed" if leg.computed else "NEEDS YOUR JUDGEMENT"
        lines.append(f"- {leg.name}: {leg.score}/5 — {leg.basis} [{marker}]")
    lines.append(f"- Provisional total: {a.uni.total}/25")
    lines.append("")

    if a.catalyst.available and a.catalyst.fundamentals:
        f = a.catalyst.fundamentals
        lines.append("### Fundamentals")
        rows = [
            (
                "Market cap",
                f"{f.market_cap / 1e9:,.2f}B {f.currency or ''}".strip()
                if f.market_cap
                else None,
            ),
            ("Sector / industry", " / ".join(x for x in (f.sector, f.industry) if x) or None),
            ("Next earnings", f"{f.next_earnings:%d %b %Y}" if f.next_earnings else None),
            ("EPS (fwd / trailing)", _pair(f.eps_forward, f.eps_trailing)),
            ("P/E (fwd / trailing)", _pair(f.forward_pe, f.trailing_pe)),
            ("PEG", f"{f.peg_ratio:.2f}" if f.peg_ratio is not None else None),
            ("Revenue growth", _pct(f.revenue_growth, signed=True)),
            ("Earnings growth", _pct(f.earnings_growth, signed=True)),
            ("Net margin", _pct(f.profit_margin)),
            (
                "Analyst targets",
                f"{money(f.target_low)} / {money(f.target_mean)} / {money(f.target_high)}"
                f" ({f.analyst_count} analysts)" if f.target_mean else None,
            ),
            ("Recommendation", f.recommendation),
            ("Insider holding", _pct(f.held_percent_insiders)),
            (
                "Insider net shares (6m)",
                f"{f.insider_net_shares_6m:,.0f}"
                if f.insider_net_shares_6m is not None
                else None,
            ),
            ("Short % float", _pct(f.short_percent_float)),
            ("Beta", f"{f.beta:.2f}" if f.beta is not None else None),
        ]
        for label, value in rows:
            lines.append(f"- {label}: {value or 'not available'}")
        lines.append("")

    if a.options.available:
        lines.append("### Options chain")
        o = a.options
        lines.append(f"- Put/call open interest: {_num(o.put_call_oi, 2)}")
        lines.append(f"- Put/call volume: {_num(o.put_call_volume, 2)}")
        lines.append(
            f"- Max pain: {money(o.max_pain)}"
            + (f" (expiry {o.max_pain_expiry:%d %b %Y})" if o.max_pain_expiry else "")
        )
        lines.append(f"- ATM implied vol: {_pct(o.atm_iv)}")
        lines.append(f"- 30-day realised vol: {_pct(o.realised_vol)}")
        lines.append(
            f"- IV vs realised: {_num(o.iv_premium, 2)}x"
            + (
                f" · realised vol sits in the "
                f"{o.realised_vol_percentile:.0f}th percentile of the year"
                if o.realised_vol_percentile is not None
                else ""
            )
        )
        lines.append(
            "- IV RANK IS NOT AVAILABLE: no free source publishes a year of implied "
            "vol history. Do not state one; use the realised-vol comparison above."
        )
        lines.append(f"- Catalyst IV expansion: {o.catalyst_expansion}")
        for item in o.unusual:
            lines.append(f"- Unusual activity: {item}")
        lines.append("")

    lines.append("### Exhaustion override")
    lines.append(f"- {a.exhaustion.summary}")
    lines.append(
        "- Rule: two or more flags are required before a wave top may be called."
    )
    lines.append("")
    return "\n".join(lines).strip()


# --- Render helpers -----------------------------------------------------------

_MAX_HOLD_MONTHS = 2

#: Scorecard glyphs, shared by the report and the page.
_SIGNAL_ICON = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪", "n/a": "—"}


def _band(value: float, thresholds: tuple[float, float, float, float]) -> int:
    """Score 5-1 by which descending threshold ``value`` clears first."""
    best, good, fair, floor = thresholds
    if value >= best:
        return 5
    if value >= good:
        return 4
    if value >= fair:
        return 3
    return 2 if value > floor else 1


def _executive_summary(a: MkrAnalysis, money: Callable[[float | None], str]) -> str:
    """Three sentences: wave position, the level that matters, the action."""
    first = (
        f"{a.symbol} is labelled {a.wave.position}"
        + (" though the count is ambiguous" if a.wave.ambiguous else "")
        + f", trading {money(a.price)} with the board at {a.confidence}/100 confidence."
    )
    if a.confirmation.death_cross_override:
        second = (
            f"The death cross override is active — the 50 EMA "
            f"({money(a.confirmation.ema50)}) is below the 200 EMA "
            f"({money(a.confirmation.ema200)}) and price is beneath both, which voids "
            "every bullish count."
        )
    elif a.levels.stop is not None and a.levels.target_1 is not None:
        second = (
            f"The level that matters right now is {money(a.levels.stop)} — the "
            f"invalidation the whole plan hangs on — against {money(a.levels.target_1)} "
            f"as the first target"
            + (
                f", a {a.levels.reward_risk:.1f}x reward-to-risk."
                if a.levels.reward_risk
                else "."
            )
        )
    else:
        second = "Structure is too thin for a defined stop and target pair yet."
    third = f"Action: {a.entry.decision} — {a.entry.rationale}"
    return f"{first} {second} {third}"


def _framework_detail(a: MkrAnalysis, money: Callable[[float | None], str]) -> list[str]:
    """One block per framework, for the traders who read past the scorecard."""
    lines: list[str] = []

    lines.append(f"**1. Elliott Wave** — {a.wave.position}. {a.wave.reason}.")
    if a.wave.invalidation:
        lines.append(f"Invalidation {money(a.wave.invalidation)}.")
    if a.wave.w3_extended is not None:
        lines.append(f"W3 extended: {'yes' if a.wave.w3_extended else 'no'}.")
    lines.append("")

    if a.fib.swing:
        grid = " · ".join(
            f"{label} {money(level)}" for label, level in a.fib.retracements.items()
        )
        lines.append(f"**2. Fibonacci** — {grid}.")
        if a.fib.golden_pocket:
            lines.append(
                f"Golden pocket {money(a.fib.golden_pocket[0])}-{money(a.fib.golden_pocket[1])}. "
                f"Price sits at {a.fib.position_pct:.0f}% of the swing."
            )
        extensions = " · ".join(
            f"{label} {money(level)}" for label, level in a.fib.extensions.items()
        )
        lines.append(f"Extensions from the {a.fib.extension_anchor}: {extensions}.")
    else:
        lines.append("**2. Fibonacci** — no confirmed swing to measure from.")
    lines.append("")

    if a.patterns.candidates:
        for candidate in a.patterns.candidates:
            target = f" → target {money(candidate.target)}" if candidate.target else ""
            lines.append(
                f"**3. Chart Patterns** — {candidate.name} ({candidate.state}): "
                f"{candidate.detail}{target}."
            )
    else:
        lines.append("**3. Chart Patterns** — no flag, coil or triangle detected.")
    lines.append(
        f"*{', '.join(a.patterns.model_checks)} need eyes on the chart — not evaluated here.*"
    )
    lines.append("")

    rsi_line = f"**4. RSI** — daily {_num(a.rsi.daily, 1)}"
    if a.rsi.weekly is not None:
        rsi_line += f", weekly {a.rsi.weekly:.1f}"
    if a.rsi.divergence:
        rsi_line += f". {a.rsi.divergence.title()} divergence: {a.rsi.divergence_detail}"
        if a.rsi.divergence.startswith("hidden"):
            rsi_line += " — a continuation signal, not a reversal"
    lines.append(rsi_line + ".")
    lines.append("")

    if a.fvg.unfilled:
        above = a.fvg.nearest_above
        below = a.fvg.nearest_below
        parts = []
        if above:
            parts.append(
                f"nearest above {money(above.low)}-{money(above.high)} ({above.timeframe})"
            )
        if below:
            parts.append(
                f"nearest below {money(below.low)}-{money(below.high)} ({below.timeframe})"
            )
        lines.append(
            f"**5. Fair Value Gaps** — {len(a.fvg.unfilled)} unfilled; "
            + ("; ".join(parts) if parts else "none straddling spot")
            + "."
        )
    else:
        lines.append("**5. Fair Value Gaps** — none unfilled in the lookback.")
    if not a.fvg.intraday_available:
        lines.append("*4-hour gaps unavailable — intraday bars could not be loaded.*")
    lines.append("")

    flow_line = "**6. Order Flow & Volume** — "
    if a.flow.ratio is not None:
        flow_line += (
            f"last session {a.flow.ratio:.2f}x the 20-day average, volume "
            f"{a.flow.trend}, OBV {a.flow.obv_direction}"
        )
        if a.flow.climax_dates:
            recent = ", ".join(f"{d:%d %b}" for d in a.flow.climax_dates[-3:])
            flow_line += f". Climax bars: {recent}"
    else:
        flow_line += "no usable volume data"
    lines.append(flow_line + ".")
    lines.append("")

    ma = a.moving_averages
    lines.append(
        f"**7. MA Structure + S/R** — 20 EMA {money(ma.ema20)}, 50 EMA {money(ma.ema50)}, "
        f"200 SMA {money(ma.sma200)}, rolling VWAP {money(ma.vwap20)}. {_sentence(ma.cross)}."
    )
    lines.append(
        f"Resistance {' / '.join(money(x) for x in ma.resistances) or 'clear above'} · "
        f"Support {' / '.join(money(x) for x in ma.supports) or 'none below'} · "
        f"ATH {money(ma.all_time_high)}."
    )
    lines.append("")

    m = a.momentum
    lines.append(
        f"**8. Momentum** — MACD histogram {_num(m.macd_hist, 3)} and {m.macd_state}; "
        f"ADX {_num(m.adx, 0)} ({'trending' if m.trending else 'no trend'}); "
        f"%B {_num(m.percent_b, 2)} ({_num(m.sigma, 1)}σ from the mean); "
        f"ATR {money(m.atr)} ({_num(m.atr_pct, 1)}% of price)."
    )
    lines.append("")

    o = a.options
    if o.available:
        lines.append(
            f"**9. Options Flow** — "
            + (
                f"from the {o.max_pain_expiry:%d %b %Y} expiry: "
                if o.max_pain_expiry
                else ""
            )
            + f"put/call OI {_num(o.put_call_oi, 2)}, volume "
            f"{_num(o.put_call_volume, 2)}; max pain {money(o.max_pain)}; ATM IV "
            f"{_pct(o.atm_iv)} against {_pct(o.realised_vol)} realised "
            f"({_num(o.iv_premium, 2)}x). {_sentence(o.catalyst_expansion)}."
        )
        lines.append(
            "*IV rank is not shown: no free feed publishes implied-vol history, so the "
            "realised-vol comparison stands in for it.*"
        )
        for item in o.unusual:
            lines.append(f"- Unusual: {item}")
    else:
        lines.append(f"**9. Options Flow** — {o.note}.")
    lines.append("")

    if a.catalyst.available and a.catalyst.fundamentals:
        f = a.catalyst.fundamentals
        earnings = (
            f"{f.next_earnings:%d %b %Y} ({a.catalyst.days_to_earnings} days)"
            if f.next_earnings
            else "not scheduled"
        )
        lines.append(
            f"**10. Catalyst & Fundamentals** — next earnings {earnings}; forward EPS "
            f"{_num(f.eps_forward, 2)}; revenue growth "
            f"{_pct(f.revenue_growth, signed=True)}; analyst targets "
            f"{money(f.target_low)} / {money(f.target_mean)} / {money(f.target_high)}"
            + (
                f" ({a.catalyst.upside_to_mean:+.1f}% to the mean)"
                if a.catalyst.upside_to_mean is not None
                else ""
            )
            + "."
        )
        if f.insider_net_shares_6m is not None:
            direction = "buying" if f.insider_net_shares_6m > 0 else "selling"
            lines.append(
                f"Insiders net {direction} "
                f"{abs(f.insider_net_shares_6m):,.0f} shares over six months."
            )
        if a.catalyst.peg_alert:
            lines.append(
                f"**{a.catalyst.peg_alert}: PEG {f.peg_ratio:.2f}.**"
            )
    else:
        lines.append(f"**10. Catalyst & Fundamentals** — {a.catalyst.note}.")
    lines.append("")

    trend_text = " · ".join(
        f"{t.timeframe} {t.trend} ({t.detail})" for t in a.timeframes.trends
    )
    verdict = (
        "all aligned — high conviction"
        if a.timeframes.aligned
        else "mixed — wait for alignment"
    )
    lines.append(f"**11. Multi-Timeframe** — {trend_text}. {verdict}.")
    lines.append("")

    lines.append(f"**12. UNI Score** — {a.uni.total}/25.")
    for leg in a.uni.legs:
        suffix = "" if leg.computed else " *(placeholder — needs judgement)*"
        lines.append(f"- {leg.name}: {leg.score}/5 — {leg.basis}{suffix}")
    lines.append("")

    c = a.confirmation
    lines.append(
        f"**13. Technical Confirmation** — 20 EMA {money(c.ema20)} / 50 EMA "
        f"{money(c.ema50)} / 200 EMA {money(c.ema200)}; RSI-14 {_num(c.rsi14, 1)}; "
        f"MACD histogram {c.macd_trend}; TRIX {_num(c.trix, 1)}bp and {c.trix_slope}."
    )
    if c.death_cross_override:
        lines.append(
            "**DEATH CROSS OVERRIDE ACTIVE — all bullish wave counts are invalidated.**"
        )
    lines.append("")

    r = a.rotation
    lines.append(
        f"**14. Capital Rotation** — enter {r.entry_size_pct:.0f}% of the intended "
        f"position. {r.scale_note} Never below a {r.cash_floor_pct:.0f}% cash floor."
    )
    if r.reentry_zone:
        lines.append(
            f"Re-entry zone after profit-taking: {money(r.reentry_zone[0])}-"
            f"{money(r.reentry_zone[1])}."
        )
    lines.append("")
    lines.append(f"**Exhaustion override** — {a.exhaustion.summary}.")
    return lines


def _avoid_condition(a: MkrAnalysis, money: Callable[[float | None], str]) -> str:
    if a.confirmation.death_cross_override:
        return "the death cross stays in force — already avoid"
    if a.levels.stop is not None:
        return f"daily close below {money(a.levels.stop)}"
    if a.moving_averages.ema50:
        return f"daily close below the 50 EMA at {money(a.moving_averages.ema50)}"
    return "structure breaks down"


def _sigma_text(leaps: LeapsPlan, horizon: str) -> str:
    if not leaps.implied_vol:
        return "n/a"
    days = 30.0 if horizon == "30d" else 60.0
    return f"{leaps.implied_vol * math.sqrt(days / 365.0) * 100:.1f}%"


def _sentence(text: str) -> str:
    """Upper-case the first character only.

    ``str.capitalize`` lower-cases everything after it, which turns "Earnings
    17 Nov" into "Earnings 17 nov".
    """
    return text[:1].upper() + text[1:] if text else text


def _paths(a: MkrAnalysis) -> int:
    for result in (a.monte_carlo_neutral, a.monte_carlo_thesis):
        if result:
            return result.paths
    return 0


def _horizon(a: MkrAnalysis) -> int:
    for result in (a.monte_carlo_neutral, a.monte_carlo_thesis):
        if result:
            return result.horizon_days
    return 0


#: Currency codes to the symbol traders actually write.
_CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "GBp": "GBp ", "JPY": "\u00a5",
    "CNY": "\u00a5", "HKD": "HK$", "AUD": "A$", "CAD": "C$", "CHF": "CHF ",
    "SEK": "SEK ", "KRW": "\u20a9", "TWD": "NT$", "INR": "\u20b9", "SGD": "S$",
}


def money_formatter(
    reference: float, currency: str | None = None
) -> Callable[[float | None], str]:
    """Pick a decimal precision and a currency mark, then stick to both.

    A 4-unit stock and a 4,000-point index need different precision; choosing
    once per report keeps every level comparable. The currency comes from the
    security's own reference data — stamping "$" on an Amsterdam listing is the
    kind of error that reads as a rounding difference until it costs money. When
    the currency is unknown the numbers are printed bare rather than guessed.
    """
    decimals = 4 if reference < 1 else 3 if reference < 10 else 2
    if currency is None:
        mark = ""
    else:
        mark = _CURRENCY_SYMBOLS.get(currency, f"{currency} ")

    def render(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{mark}{value:,.{decimals}f}"

    return render


def _pair(first: float | None, second: float | None) -> str | None:
    if first is None and second is None:
        return None
    return f"{_num(first, 2)} / {_num(second, 2)}"


def _num(value: float | None, decimals: int) -> str:
    return "n/a" if value is None else f"{value:,.{decimals}f}"


def _pct(value: float | None, signed: bool = False) -> str | None:
    """A ratio as a percentage, or None so callers can say "not available"."""
    if value is None:
        return None
    return f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"
