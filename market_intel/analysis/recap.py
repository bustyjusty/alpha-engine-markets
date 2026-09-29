"""Pure computation for the global market recap.

No I/O and no Streamlit: everything here takes price series plus a clock and
returns plain dataclasses, which is what makes the recap testable without a
network or a database.

The module solves three problems the raw price data does not:

1. **Which session am I looking at?** At 11:00 in Singapore, APAC is mid-session
   while London and New York last printed the day before. A recap that stamps
   every region "today" is quietly wrong, so :func:`session_state` classifies
   each region and the UI labels the numbers accordingly.
2. **What counts as a move?** Yields move in basis points, indices in percent.
   :func:`compute_move` reads :class:`~market_intel.universe.QuoteKind` and fills
   whichever field is meaningful.
3. **What actually mattered?** :func:`rank_by_magnitude` and
   :func:`block_stats` turn eighty rows into the handful worth reading first.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from market_intel.universe import AssetClass, Instrument, QuoteKind, Region


class SessionState(str, Enum):
    """Whether a region's cash market is trading right now.

    Attributes:
        OPEN: Cash session is live; the quote is an intraday print.
        CLOSED: Weekday, but outside session hours; the quote is a prior close.
        WEEKEND: Saturday or Sunday in the region.
    """

    OPEN = "open"
    CLOSED = "closed"
    WEEKEND = "weekend"


#: Approximate cash-equity session windows in UTC (start_hour, end_hour).
#:
#: These are display hints, not a trading calendar: they ignore daylight-saving
#: shifts, holidays and lunch breaks. They exist so the UI can say "live" vs
#: "last close" — never to decide whether an order could be worked.
_SESSION_HOURS_UTC: dict[Region, tuple[float, float]] = {
    Region.APAC: (0.0, 8.0),  # Tokyo open through Hong Kong close
    Region.UK: (7.0, 15.5),  # LSE 08:00-16:30 London
    Region.EUROPE: (7.0, 15.5),  # Frankfurt/Paris 09:00-17:30 CET
    Region.US: (13.5, 20.0),  # NYSE 09:30-16:00 New York
    Region.GLOBAL: (0.0, 24.0),  # Futures trade nearly around the clock
}


def session_state(region: Region, now_utc: dt.datetime) -> SessionState:
    """Classify a region's session at ``now_utc``.

    Args:
        region: Region to classify.
        now_utc: Current time. Naive values are assumed to already be UTC.

    Returns:
        :attr:`SessionState.WEEKEND` on Saturday/Sunday, otherwise
        :attr:`SessionState.OPEN` inside the region's window and
        :attr:`SessionState.CLOSED` outside it.
    """
    if now_utc.tzinfo is not None:
        now_utc = now_utc.astimezone(dt.timezone.utc).replace(tzinfo=None)
    if now_utc.weekday() >= 5:
        return SessionState.WEEKEND

    start, end = _SESSION_HOURS_UTC[region]
    hour = now_utc.hour + now_utc.minute / 60.0
    return SessionState.OPEN if start <= hour < end else SessionState.CLOSED


def session_label(region: Region, state: SessionState) -> str:
    """Return a short human description of what the region's numbers represent."""
    if state is SessionState.OPEN:
        return "Live - session in progress"
    if state is SessionState.WEEKEND:
        return "Weekend - showing Friday's close"
    return "Closed - showing last completed session"


#: Above this level an FX pair is quoted in whole units rather than pips
#: (USD/JPY at 158.97, USD/KRW at 1381.26), so four decimals are noise.
_FX_WHOLE_UNIT_THRESHOLD = 20.0


def format_level(value: float, quote: QuoteKind) -> str:
    """Format a price/level at the precision its market conventionally uses.

    Args:
        value: The level to render.
        quote: How the instrument is quoted.

    Returns:
        Yields as a percentage to two decimals, FX to four decimals (two for
        pairs quoted in whole units), and everything else to two decimals with
        thousands separators.
    """
    if quote is QuoteKind.YIELD:
        return f"{value:,.2f}%"
    if quote is QuoteKind.FX:
        if abs(value) >= _FX_WHOLE_UNIT_THRESHOLD:
            return f"{value:,.2f}"
        return f"{value:,.4f}"
    return f"{value:,.2f}"


def level_format_for(moves: list["InstrumentMove"]) -> str:
    """Return the printf format a table of ``moves`` should use for levels.

    Streamlit's column config takes one format for the whole column, so a block
    of sub-unit FX pairs gets four decimals and everything else gets two.
    """
    if moves and all(
        move.instrument.quote is QuoteKind.FX
        and abs(move.last) < _FX_WHOLE_UNIT_THRESHOLD
        for move in moves
    ):
        return "%.4f"
    return "%.2f"


@dataclass(frozen=True, slots=True)
class InstrumentMove:
    """One instrument's move, computed from its close series.

    Attributes:
        instrument: The universe entry this move describes.
        last: Most recent close (or intraday print) available.
        previous: The close before ``last``.
        change: Absolute change in quote units.
        pct_change: Percentage change versus ``previous``.
        bp_change: Change in basis points; only set for yield quotes.
        ret_5d: Trailing 5-observation return, or None if history is short.
        ret_1m: Trailing 21-observation return, or None if history is short.
        ret_ytd: Return from the first observation of the current year.
        as_of: Date of ``last``.
        observations: Number of closes the calculation saw.
        partial_session: The latest bar traded on unusually thin volume and
            is probably an in-progress session, so its move is not yet
            comparable with the completed sessions it is ranked against.
    """

    instrument: Instrument
    last: float
    previous: float
    change: float
    pct_change: float
    bp_change: float | None
    ret_5d: float | None
    ret_1m: float | None
    ret_ytd: float | None
    as_of: dt.date
    observations: int
    partial_session: bool = False

    @property
    def is_rate(self) -> bool:
        """Whether this is a yield or spread quoted in percent."""
        return self.instrument.quote is QuoteKind.YIELD

    @property
    def magnitude(self) -> float:
        """Absolute percentage move, used for ranking price-quoted instruments.

        Meaningless for rates and spreads, which is why
        :func:`rank_by_magnitude` excludes them: a spread that moves from 0.46
        to 0.50 is up 8.7% but only 4bp, and ranking it against an equity index
        would put a trivial rates move at the top of the movers list.
        """
        return abs(self.pct_change)

    @property
    def rate_magnitude(self) -> float:
        """Absolute move in basis points; zero for non-rate instruments."""
        return abs(self.bp_change) if self.bp_change is not None else 0.0

    @property
    def display_change(self) -> str:
        """The move formatted the way this asset class is normally quoted."""
        if self.bp_change is not None:
            return f"{self.bp_change:+.1f} bp"
        return f"{self.pct_change:+.2f}%"

    @property
    def display_level(self) -> str:
        """The last level formatted at the precision its market is quoted in.

        Generic formats do not survive a cross-asset table: ``%.4g`` renders the
        Nikkei as ``6.601e+04`` and rounds USD/JPY to ``159``, throwing away the
        pips a trader is actually watching. Precision follows the instrument's
        own convention instead.
        """
        return format_level(self.last, self.instrument.quote)

    def is_stale(self, today: dt.date, max_age_days: int = 4) -> bool:
        """Whether ``as_of`` is old enough to warrant a staleness warning.

        The default tolerance spans a long weekend plus a public holiday, which
        is normal for a market that simply was not open.
        """
        return (today - self.as_of).days > max_age_days


#: A session whose volume is below this share of the recent median is treated
#: as still in progress rather than a completed day.
_PARTIAL_VOLUME_SHARE = 0.35


def is_partial_session(volumes: pd.Series | None) -> bool:
    """Whether the latest bar looks like an incomplete trading session.

    A market that opened minutes ago prints a real bar on tiny volume. Compared
    against a full prior session it can show an enormous "move" that no one
    traded - grains an hour into the open are the classic case. Comparing the
    last bar's volume with the median of the preceding ten separates a thin
    partial session from a genuine one.

    Returns ``False`` when volume is unavailable (FX, indices and FRED series
    carry none), since absence of data is not evidence of a partial session.
    """
    if volumes is None:
        return False
    series = pd.to_numeric(volumes, errors="coerce").dropna()
    if len(series) < 4:
        return False

    last = float(series.iloc[-1])
    # Only prior bars that actually traded set the baseline.
    recent = series.iloc[-11:-1]
    recent = recent[recent > 0]
    if recent.empty:
        return False
    median = float(recent.median())
    if median <= 0:
        return False

    # A bar reporting zero volume in a market that normally trades is not a
    # thin session, it is a bad print - the provider occasionally serves one,
    # and it can carry a wildly wrong close. Filtering zeros out before this
    # comparison would hide exactly the case worth catching.
    if last <= 0:
        return True
    return last < median * _PARTIAL_VOLUME_SHARE


def compute_move(
    instrument: Instrument,
    closes: pd.Series,
    volumes: pd.Series | None = None,
) -> InstrumentMove | None:
    """Compute one instrument's move from a date-indexed close series.

    Args:
        instrument: Universe entry describing how to read the quote.
        closes: Closes indexed by date, oldest first. NaNs are dropped.
        volumes: Matching volume series, used only to detect a partial session.

    Returns:
        An :class:`InstrumentMove`, or ``None`` when fewer than two usable
        closes exist or the previous close is zero (no meaningful percentage).
    """
    series = pd.to_numeric(closes, errors="coerce").dropna()
    if len(series) < 2:
        return None

    last = float(series.iloc[-1])
    previous = float(series.iloc[-2])
    if previous == 0:
        return None

    change = last - previous
    pct_change = change / previous * 100.0
    # A yield quoted in percent moves in basis points: 4.70 -> 4.74 is +4bp.
    bp_change = change * 100.0 if instrument.quote is QuoteKind.YIELD else None

    def window_return(periods: int) -> float | None:
        if len(series) <= periods:
            return None
        base = float(series.iloc[-1 - periods])
        return (last / base - 1.0) * 100.0 if base else None

    as_of = _as_date(series.index[-1])
    ytd = _ytd_return(series, last, as_of.year)

    return InstrumentMove(
        instrument=instrument,
        last=last,
        previous=previous,
        change=change,
        pct_change=pct_change,
        bp_change=bp_change,
        ret_5d=window_return(5),
        ret_1m=window_return(21),
        ret_ytd=ytd,
        as_of=as_of,
        observations=len(series),
        partial_session=is_partial_session(volumes),
    )


def _ytd_return(series: pd.Series, last: float, year: int) -> float | None:
    """Return percentage change from the first observation of ``year``."""
    this_year = [
        float(value)
        for stamp, value in series.items()
        if _as_date(stamp).year == year and float(value) != 0
    ]
    if len(this_year) < 2:
        return None
    return (last / this_year[0] - 1.0) * 100.0


def _as_date(stamp: object) -> dt.date:
    """Coerce a pandas index entry to a plain date."""
    if isinstance(stamp, dt.datetime):
        return stamp.date()
    if isinstance(stamp, dt.date):
        return stamp
    return pd.Timestamp(stamp).date()


def rank_by_magnitude(moves: list[InstrumentMove], limit: int = 5) -> list[InstrumentMove]:
    """Return the ``limit`` largest absolute movers among price-quoted instruments.

    Rates and spreads are excluded because percentage change is not comparable
    across units - use :func:`rank_rates_by_bp` for those. Partial sessions are
    excluded too: a market twenty minutes into its open can print a huge move on
    negligible volume, and it would otherwise top the board every morning.
    """
    priced = [
        move for move in moves if not move.is_rate and not move.partial_session
    ]
    return sorted(priced, key=lambda move: move.magnitude, reverse=True)[:limit]


def rank_rates_by_bp(moves: list[InstrumentMove], limit: int = 5) -> list[InstrumentMove]:
    """Return the ``limit`` largest rate moves in basis points, biggest first."""
    rates = [move for move in moves if move.bp_change is not None]
    return sorted(rates, key=lambda move: move.rate_magnitude, reverse=True)[:limit]


def top_gainers(moves: list[InstrumentMove], limit: int = 5) -> list[InstrumentMove]:
    """Return the ``limit`` strongest positive price moves, best first."""
    ups = [
        move
        for move in moves
        if move.pct_change > 0 and not move.is_rate and not move.partial_session
    ]
    return sorted(ups, key=lambda move: move.pct_change, reverse=True)[:limit]


def top_losers(moves: list[InstrumentMove], limit: int = 5) -> list[InstrumentMove]:
    """Return the ``limit`` weakest price moves, worst first."""
    downs = [
        move
        for move in moves
        if move.pct_change < 0 and not move.is_rate and not move.partial_session
    ]
    return sorted(downs, key=lambda move: move.pct_change)[:limit]


@dataclass(frozen=True, slots=True)
class BlockStats:
    """Breadth summary for one region/asset-class block.

    Attributes:
        count: Instruments with a computable move.
        advancers: How many closed higher.
        decliners: How many closed lower.
        unchanged: How many were exactly flat.
        average_pct: Mean percentage move across the block.
        widest: The single largest absolute mover, if any.
    """

    count: int
    advancers: int
    decliners: int
    unchanged: int
    average_pct: float
    widest: InstrumentMove | None


def block_stats(moves: list[InstrumentMove]) -> BlockStats:
    """Summarise breadth and the average move across a block.

    Breadth counts every instrument, but the average and the widest mover are
    computed over price-quoted instruments only: a spread's percentage change
    is not on the same scale as an index's and would dominate both.
    """
    if not moves:
        return BlockStats(0, 0, 0, 0, 0.0, None)
    advancers = sum(1 for move in moves if move.pct_change > 0)
    decliners = sum(1 for move in moves if move.pct_change < 0)
    priced = [move for move in moves if not move.is_rate]
    average = sum(move.pct_change for move in priced) / len(priced) if priced else 0.0
    return BlockStats(
        count=len(moves),
        advancers=advancers,
        decliners=decliners,
        unchanged=len(moves) - advancers - decliners,
        average_pct=average,
        widest=max(priced, key=lambda move: move.magnitude) if priced else None,
    )


def group_moves(
    moves: list[InstrumentMove],
) -> dict[Region, dict[AssetClass, list[InstrumentMove]]]:
    """Bucket moves into ``region -> asset class -> moves``, order preserved."""
    grouped: dict[Region, dict[AssetClass, list[InstrumentMove]]] = {}
    for move in moves:
        region_bucket = grouped.setdefault(move.instrument.region, {})
        region_bucket.setdefault(move.instrument.asset_class, []).append(move)
    return grouped
