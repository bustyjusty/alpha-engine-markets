"""Fundamentals, option chain and intraday bars for single-name analysis.

The core :class:`~market_intel.providers.base.MarketDataProvider` interface is
deliberately narrow — reference data and daily OHLCV — because that is all the
cross-asset pages need, and every extra abstract method is another thing a new
adapter has to implement. The MKR framework needs three things beyond it:
fundamentals (PEG, analyst targets, earnings date), the option chain (put/call,
max pain, implied vol) and intraday bars (4-hour fair value gaps).

Rather than widen the shared interface for one page, this module adds a
separate optional provider. Services ask for it, catch failure, and carry on:
every one of these three feeds can be absent and the analysis still produces a
report with those sections marked unavailable. Nothing here is ever inferred —
a missing field stays ``None`` rather than being filled with a plausible guess.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import yfinance as yf

from market_intel.exceptions import DataNotFoundError, ProviderError

logger = logging.getLogger(__name__)

#: Cap on how many expiries are pulled — each one is a separate HTTP round trip.
DEFAULT_MAX_EXPIRIES = 6
#: Chain statistics ignore expiries closer than this; 0DTE flow is noise.
STATS_MIN_DTE = 7
#: Days-to-expiry buckets the frameworks need one expiry from each of.
EXPIRY_BUCKETS: tuple[int, ...] = (0, 7, 21, 45, 180, 365)


def select_expiries(
    expiries: list[str], limit: int = DEFAULT_MAX_EXPIRIES, today: dt.date | None = None
) -> list[str]:
    """Pick a spread of expiries covering every maturity the frameworks use.

    For each bucket in :data:`EXPIRY_BUCKETS` the nearest expiry at or beyond it
    is taken, and the furthest listed expiry is always included so a LEAPS leg
    exists. Duplicates collapse, so a name with only three expiries costs three
    requests rather than six.
    """
    today = today or dt.date.today()
    parsed: list[tuple[int, str]] = []
    for value in expiries:
        try:
            parsed.append(((dt.date.fromisoformat(value) - today).days, value))
        except ValueError:
            continue
    if not parsed:
        return expiries[:limit]
    parsed.sort()

    chosen: list[str] = []
    for bucket in EXPIRY_BUCKETS:
        match = next((value for days, value in parsed if days >= bucket), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    furthest = parsed[-1][1]
    if furthest not in chosen:
        chosen.append(furthest)
    return chosen[:limit] if limit else chosen


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Company-level facts the catalyst framework needs.

    Every field is optional. Yahoo omits most of them for indices, futures and
    many non-US listings, and a missing value must read as "not available"
    rather than zero.
    """

    symbol: str
    name: str | None = None
    currency: str | None = None
    market_cap: float | None = None
    peg_ratio: float | None = None
    forward_pe: float | None = None
    trailing_pe: float | None = None
    eps_forward: float | None = None
    eps_trailing: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    profit_margin: float | None = None
    target_low: float | None = None
    target_mean: float | None = None
    target_high: float | None = None
    analyst_count: int | None = None
    recommendation: str | None = None
    held_percent_insiders: float | None = None
    held_percent_institutions: float | None = None
    short_percent_float: float | None = None
    beta: float | None = None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    next_earnings: dt.date | None = None
    insider_net_shares_6m: float | None = None
    sector: str | None = None
    industry: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.next_earnings is not None:
            payload["next_earnings"] = self.next_earnings.isoformat()
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Fundamentals:
        data = dict(payload)
        stamp = data.get("next_earnings")
        data["next_earnings"] = dt.date.fromisoformat(stamp) if stamp else None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """One listed contract."""

    expiry: dt.date
    strike: float
    kind: str  # "call" | "put"
    bid: float | None
    ask: float | None
    last: float | None
    implied_vol: float | None
    open_interest: int | None
    volume: int | None

    @property
    def mid(self) -> float | None:
        """Mid price, falling back to the last trade when the book is empty."""
        if self.bid is not None and self.ask is not None and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last


@dataclass(frozen=True, slots=True)
class OptionsSnapshot:
    """Chain-level statistics plus the raw quotes the service needs.

    ``iv_rank`` is deliberately absent. A true IV rank needs a year of implied
    volatility history, which no free source provides; the service reports ATM
    IV against *realised* volatility instead and says so.
    """

    symbol: str
    as_of: dt.date
    expiries: tuple[dt.date, ...] = ()
    quotes: tuple[OptionQuote, ...] = ()
    put_call_oi: float | None = None
    put_call_volume: float | None = None
    max_pain: float | None = None
    max_pain_expiry: dt.date | None = None
    atm_iv: float | None = None
    unusual: tuple[str, ...] = ()

    def for_expiry(self, expiry: dt.date, kind: str) -> list[OptionQuote]:
        return [
            quote
            for quote in self.quotes
            if quote.expiry == expiry and quote.kind == kind
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "expiries": [expiry.isoformat() for expiry in self.expiries],
            "quotes": [
                {**asdict(quote), "expiry": quote.expiry.isoformat()}
                for quote in self.quotes
            ],
            "put_call_oi": self.put_call_oi,
            "put_call_volume": self.put_call_volume,
            "max_pain": self.max_pain,
            "max_pain_expiry": (
                self.max_pain_expiry.isoformat() if self.max_pain_expiry else None
            ),
            "atm_iv": self.atm_iv,
            "unusual": list(self.unusual),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OptionsSnapshot:
        pain_expiry = payload.get("max_pain_expiry")
        return cls(
            symbol=payload["symbol"],
            as_of=dt.date.fromisoformat(payload["as_of"]),
            expiries=tuple(
                dt.date.fromisoformat(value) for value in payload.get("expiries", [])
            ),
            quotes=tuple(
                OptionQuote(
                    expiry=dt.date.fromisoformat(quote["expiry"]),
                    strike=quote["strike"],
                    kind=quote["kind"],
                    bid=quote["bid"],
                    ask=quote["ask"],
                    last=quote["last"],
                    implied_vol=quote["implied_vol"],
                    open_interest=quote["open_interest"],
                    volume=quote["volume"],
                )
                for quote in payload.get("quotes", [])
            ),
            put_call_oi=payload.get("put_call_oi"),
            put_call_volume=payload.get("put_call_volume"),
            max_pain=payload.get("max_pain"),
            max_pain_expiry=dt.date.fromisoformat(pain_expiry) if pain_expiry else None,
            atm_iv=payload.get("atm_iv"),
            unusual=tuple(payload.get("unusual", [])),
        )


class YFinanceFundamentalsProvider:
    """Fundamentals, options and intraday bars from Yahoo Finance."""

    name = "yfinance"

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Company facts for one symbol.

        Raises:
            DataNotFoundError: Yahoo has no quote for the symbol.
            ProviderError: The request itself failed.
        """
        ticker = yf.Ticker(symbol)
        try:
            info: dict[str, Any] = ticker.get_info()
        except Exception as exc:  # yfinance raises assorted internal errors
            raise ProviderError(
                f"yfinance fundamentals request failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc
        if not info or info.get("quoteType") in (None, "NONE"):
            raise DataNotFoundError(
                f"No fundamentals found for '{symbol}'", provider=self.name
            )

        return Fundamentals(
            symbol=symbol.strip().upper(),
            name=info.get("longName") or info.get("shortName"),
            currency=info.get("currency"),
            market_cap=_as_float(info.get("marketCap")),
            peg_ratio=_as_float(
                info.get("trailingPegRatio") or info.get("pegRatio")
            ),
            forward_pe=_as_float(info.get("forwardPE")),
            trailing_pe=_as_float(info.get("trailingPE")),
            eps_forward=_as_float(info.get("forwardEps")),
            eps_trailing=_as_float(info.get("trailingEps")),
            revenue_growth=_as_float(info.get("revenueGrowth")),
            earnings_growth=_as_float(
                info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
            ),
            profit_margin=_as_float(info.get("profitMargins")),
            target_low=_as_float(info.get("targetLowPrice")),
            target_mean=_as_float(info.get("targetMeanPrice")),
            target_high=_as_float(info.get("targetHighPrice")),
            analyst_count=_as_int(info.get("numberOfAnalystOpinions")),
            recommendation=info.get("recommendationKey"),
            held_percent_insiders=_as_float(info.get("heldPercentInsiders")),
            held_percent_institutions=_as_float(info.get("heldPercentInstitutions")),
            short_percent_float=_as_float(info.get("shortPercentOfFloat")),
            beta=_as_float(info.get("beta")),
            fifty_two_week_low=_as_float(info.get("fiftyTwoWeekLow")),
            fifty_two_week_high=_as_float(info.get("fiftyTwoWeekHigh")),
            next_earnings=self._next_earnings(ticker, info),
            insider_net_shares_6m=self._insider_net(ticker),
            sector=info.get("sector"),
            industry=info.get("industry"),
        )

    def get_options(
        self, symbol: str, max_expiries: int = DEFAULT_MAX_EXPIRIES
    ) -> OptionsSnapshot:
        """Option chain statistics across a deliberate spread of expiries.

        Taking the nearest N expiries is the obvious approach and the wrong one:
        on a liquid name like NVDA the first six are all weeklies inside a
        month, which leaves the wheel with nothing in its 21-60 day window and
        the LEAPS leg with nothing at all. :func:`select_expiries` instead picks
        one expiry per maturity bucket the frameworks actually use.

        Raises:
            DataNotFoundError: The symbol has no listed options.
            ProviderError: The request failed.
        """
        ticker = yf.Ticker(symbol)
        try:
            raw_expiries = list(ticker.options or ())
        except Exception as exc:
            raise ProviderError(
                f"yfinance options request failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc
        if not raw_expiries:
            raise DataNotFoundError(
                f"No listed options for '{symbol}'", provider=self.name
            )

        wanted = select_expiries(raw_expiries, limit=max_expiries)

        quotes: list[OptionQuote] = []
        loaded: list[dt.date] = []
        for stamp in wanted:
            try:
                chain = ticker.option_chain(stamp)
            except Exception as exc:  # one bad expiry must not kill the chain
                logger.warning("Option chain %s %s failed: %s", symbol, stamp, exc)
                continue
            expiry = dt.date.fromisoformat(stamp)
            loaded.append(expiry)
            quotes.extend(_frame_to_quotes(chain.calls, expiry, "call"))
            quotes.extend(_frame_to_quotes(chain.puts, expiry, "put"))

        if not quotes:
            raise DataNotFoundError(
                f"Option chain for '{symbol}' came back empty", provider=self.name
            )

        spot = _as_float(_spot_from_info(ticker))
        # Chain statistics come from the first expiry at least a week out. A
        # zero-day expiry has a put/call ratio dominated by same-day scalping
        # and a max pain that means nothing by tomorrow.
        today = dt.date.today()
        settled = [
            expiry for expiry in sorted(loaded) if (expiry - today).days >= STATS_MIN_DTE
        ]
        front = settled[0] if settled else (min(loaded) if loaded else None)
        front_quotes = [quote for quote in quotes if quote.expiry == front]
        return OptionsSnapshot(
            symbol=symbol.strip().upper(),
            as_of=dt.date.today(),
            expiries=tuple(sorted(loaded)),
            quotes=tuple(quotes),
            put_call_oi=_put_call(front_quotes, "open_interest"),
            put_call_volume=_put_call(front_quotes, "volume"),
            max_pain=max_pain(front_quotes),
            max_pain_expiry=front,
            atm_iv=atm_implied_vol(front_quotes, spot) if spot else None,
            unusual=tuple(unusual_activity(front_quotes)),
        )

    def get_intraday(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Hourly bars resampled to 4 hours, for intraday fair value gaps.

        Yahoo only serves ~2 years of hourly data and nothing finer for free;
        60 days is plenty for the gaps that are still open.

        Raises:
            DataNotFoundError: No intraday history available.
            ProviderError: The request failed.
        """
        try:
            frame = yf.Ticker(symbol).history(
                period=f"{days}d", interval="1h", auto_adjust=False
            )
        except Exception as exc:
            raise ProviderError(
                f"yfinance intraday request failed for '{symbol}': {exc}",
                provider=self.name,
            ) from exc
        if frame.empty:
            raise DataNotFoundError(
                f"No intraday bars for '{symbol}'", provider=self.name
            )

        frame = frame.rename(columns=str.lower)
        aggregation = {
            column: how
            for column, how in (
                ("open", "first"), ("high", "max"),
                ("low", "min"), ("close", "last"), ("volume", "sum"),
            )
            if column in frame.columns
        }
        resampled = frame.resample("4h").agg(aggregation).dropna(subset=["close"])
        resampled.index.name = "date"
        return resampled

    # --- internals ------------------------------------------------------------

    def _next_earnings(self, ticker: Any, info: dict[str, Any]) -> dt.date | None:
        """Next scheduled earnings date, from the calendar or the info blob."""
        try:
            frame = ticker.get_earnings_dates(limit=8)
        except Exception:  # noqa: BLE001 - optional enrichment
            frame = None
        today = dt.date.today()
        if frame is not None and not frame.empty:
            upcoming = [
                stamp.date() if hasattr(stamp, "date") else stamp
                for stamp in frame.index
            ]
            future = sorted(stamp for stamp in upcoming if stamp >= today)
            if future:
                return future[0]
        timestamp = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if timestamp:
            try:
                return dt.datetime.fromtimestamp(int(timestamp), dt.UTC).date()
            except (ValueError, OSError, OverflowError):
                return None
        return None

    def _insider_net(self, ticker: Any) -> float | None:
        """Net insider shares transacted in the last six months, if published."""
        try:
            frame = ticker.insider_transactions
        except Exception:  # noqa: BLE001 - optional enrichment
            return None
        if frame is None or getattr(frame, "empty", True):
            return None
        columns = {str(name).lower(): name for name in frame.columns}
        shares_column = columns.get("shares")
        text_column = columns.get("text") or columns.get("transaction")
        if shares_column is None:
            return None
        cutoff = dt.date.today() - dt.timedelta(days=182)
        date_column = columns.get("start date") or columns.get("date")
        total = 0.0
        for _, row in frame.iterrows():
            if date_column is not None:
                stamp = row[date_column]
                stamp = stamp.date() if hasattr(stamp, "date") else stamp
                if isinstance(stamp, dt.date) and stamp < cutoff:
                    continue
            shares = _as_float(row[shares_column])
            if shares is None:
                continue
            description = str(row[text_column]).lower() if text_column else ""
            sign = -1.0 if ("sale" in description or "sold" in description) else 1.0
            total += sign * shares
        return total


# --- Chain analytics ----------------------------------------------------------


def max_pain(quotes: list[OptionQuote]) -> float | None:
    """The strike at which total option-holder payout is smallest.

    Computed over open interest at one expiry: for each listed strike, sum what
    every in-the-money call and put would pay if the underlying settled there.
    """
    strikes = sorted({quote.strike for quote in quotes if quote.open_interest})
    if len(strikes) < 3:
        return None
    best_strike, best_payout = None, math.inf
    for settle in strikes:
        payout = 0.0
        for quote in quotes:
            interest = quote.open_interest or 0
            if not interest:
                continue
            if quote.kind == "call" and settle > quote.strike:
                payout += (settle - quote.strike) * interest
            elif quote.kind == "put" and settle < quote.strike:
                payout += (quote.strike - settle) * interest
        if payout < best_payout:
            best_strike, best_payout = settle, payout
    return best_strike


def atm_implied_vol(quotes: list[OptionQuote], spot: float) -> float | None:
    """Average implied vol of the call and put closest to spot."""
    candidates = [
        quote for quote in quotes if quote.implied_vol and quote.implied_vol > 0
    ]
    if not candidates or spot <= 0:
        return None
    nearest = min(candidates, key=lambda quote: abs(quote.strike - spot))
    same_strike = [
        quote.implied_vol
        for quote in candidates
        if abs(quote.strike - nearest.strike) < 1e-9 and quote.implied_vol
    ]
    return sum(same_strike) / len(same_strike) if same_strike else None


def unusual_activity(quotes: list[OptionQuote], threshold: float = 3.0) -> list[str]:
    """Contracts whose day volume dwarfs their open interest."""
    flagged: list[tuple[float, str]] = []
    for quote in quotes:
        volume = quote.volume or 0
        interest = quote.open_interest or 0
        if volume < 250 or interest < 50:
            continue
        ratio = volume / interest
        if ratio >= threshold:
            flagged.append(
                (
                    ratio,
                    f"{quote.expiry:%d %b} {quote.strike:g}{quote.kind[0].upper()} "
                    f"vol {volume:,} vs OI {interest:,} ({ratio:.1f}x)",
                )
            )
    flagged.sort(reverse=True)
    return [text for _, text in flagged[:4]]


def _put_call(quotes: list[OptionQuote], attribute: str) -> float | None:
    """Put/call ratio on open interest or volume."""
    calls = sum(getattr(q, attribute) or 0 for q in quotes if q.kind == "call")
    puts = sum(getattr(q, attribute) or 0 for q in quotes if q.kind == "put")
    return puts / calls if calls else None


def _frame_to_quotes(
    frame: pd.DataFrame, expiry: dt.date, kind: str
) -> list[OptionQuote]:
    if frame is None or frame.empty:
        return []
    quotes: list[OptionQuote] = []
    for _, row in frame.iterrows():
        strike = _as_float(row.get("strike"))
        if strike is None:
            continue
        quotes.append(
            OptionQuote(
                expiry=expiry,
                strike=strike,
                kind=kind,
                bid=_as_float(row.get("bid")),
                ask=_as_float(row.get("ask")),
                last=_as_float(row.get("lastPrice")),
                implied_vol=_as_float(row.get("impliedVolatility")),
                open_interest=_as_int(row.get("openInterest")),
                volume=_as_int(row.get("volume")),
            )
        )
    return quotes


def _spot_from_info(ticker: Any) -> float | None:
    try:
        info = ticker.fast_info
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            value = getattr(info, key, None) if not isinstance(info, dict) else info.get(key)
            if value:
                return float(value)
    except Exception:  # noqa: BLE001 - optional enrichment
        return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return int(number) if number is not None else None
