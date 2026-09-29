"""Alpha Vantage adapter - a documented, licensed API, on a very small budget.

Alpha Vantage is the most *legitimate* of the free sources here: a published API
with terms of service, rather than a scrape. That makes it a good last resort
when the primary feed breaks.

**The free tier allows 25 requests per day.** The recap universe is roughly
ninety instruments, so a single full refresh would consume three and a half days
of quota. This adapter therefore refuses to be used casually:

* It tracks its own call count and stops at ``daily_budget``, raising
  :class:`RateLimitError` rather than silently burning the remaining allowance.
* It belongs **last** in :class:`~market_intel.providers.chain.ChainedMarketDataProvider`,
  where it is only reached for symbols the free, unlimited sources could not serve.

The budget counter is per-process and resets when the Streamlit server restarts,
so it is a guard rail rather than an exact ledger. Alpha Vantage itself enforces
the real limit, which this adapter surfaces as a typed error.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from market_intel.exceptions import DataNotFoundError, ProviderError, RateLimitError
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

_ENDPOINT = "https://www.alphavantage.co/query"
_TIMEOUT_SECONDS = 25
_USER_AGENT = "market-intel/0.1 (research dashboard)"

#: Universe FX symbols -> (from_symbol, to_symbol) for the FX_DAILY endpoint.
_FX_PAIRS: dict[str, tuple[str, str]] = {
    "EURUSD=X": ("EUR", "USD"),
    "GBPUSD=X": ("GBP", "USD"),
    "USDJPY=X": ("USD", "JPY"),
    "AUDUSD=X": ("AUD", "USD"),
    "NZDUSD=X": ("NZD", "USD"),
    "USDCHF=X": ("USD", "CHF"),
    "USDCAD=X": ("USD", "CAD"),
    "USDSGD=X": ("USD", "SGD"),
    "USDKRW=X": ("USD", "KRW"),
    "USDINR=X": ("USD", "INR"),
    "EURGBP=X": ("EUR", "GBP"),
    "CNY=X": ("USD", "CNY"),
}


class AlphaVantageProvider(MarketDataProvider):
    """Daily bars from Alpha Vantage, for equities/ETFs and major FX pairs.

    Args:
        api_key: Alpha Vantage key. Without one the provider reports every
            symbol as uncovered so the chain skips it cleanly.
        daily_budget: Maximum requests this process will make in its lifetime.
            Defaults to the free-tier allowance.
    """

    name = "alphavantage"

    def __init__(self, api_key: str | None = None, daily_budget: int = 25) -> None:
        self._api_key = api_key or None
        self._daily_budget = daily_budget
        self._calls_made = 0

    @property
    def calls_remaining(self) -> int:
        """Requests left in this process's self-imposed budget."""
        return max(0, self._daily_budget - self._calls_made)

    def get_security_info(self, symbol: str) -> SecurityInfo:
        """Return minimal reference data without spending a request.

        Reference data is not worth a slot from a 25-request budget, so this
        returns a stub for symbols the adapter can price and raises otherwise.
        """
        key = symbol.strip().upper()
        if key in _FX_PAIRS:
            return SecurityInfo(symbol=key, name=key, asset_type="fx")
        if _is_plain_ticker(key):
            return SecurityInfo(symbol=key, name=key, asset_type="equity")
        raise DataNotFoundError(
            f"Alpha Vantage does not cover '{symbol}'", provider=self.name
        )

    def get_daily_bars(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[PriceBar]:
        """Return daily bars for ``start``..``end`` inclusive.

        Raises:
            DataNotFoundError: No key configured, or the symbol is not one this
                adapter can request (indices and futures have no free endpoint).
            RateLimitError: The local budget or the API's own limit was hit.
            ProviderError: Any other transport or parse failure.
        """
        if not self._api_key:
            raise DataNotFoundError(
                "No Alpha Vantage API key configured", provider=self.name
            )

        key = symbol.strip().upper()
        if key in _FX_PAIRS:
            base, quote = _FX_PAIRS[key]
            params = {
                "function": "FX_DAILY",
                "from_symbol": base,
                "to_symbol": quote,
                "outputsize": "full",
            }
            series_key = "Time Series FX (Daily)"
        elif _is_plain_ticker(key):
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": key,
                "outputsize": "full",
            }
            series_key = "Time Series (Daily)"
        else:
            # Indices (^GSPC) and futures (GC=F) have no free Alpha Vantage
            # endpoint; say so plainly so the chain moves on without a request.
            raise DataNotFoundError(
                f"Alpha Vantage has no free daily endpoint for '{symbol}'",
                provider=self.name,
            )

        payload = self._request(params, symbol)
        series = payload.get(series_key)
        if not isinstance(series, dict) or not series:
            raise DataNotFoundError(
                f"Alpha Vantage returned no series for '{symbol}'", provider=self.name
            )

        bars: list[PriceBar] = []
        for raw_date, values in series.items():
            try:
                day = dt.date.fromisoformat(raw_date)
            except ValueError:
                continue
            if not start <= day <= end:
                continue
            close = _to_float(values.get("4. close"))
            if close is None:
                continue
            bars.append(
                PriceBar(
                    date=day,
                    close=close,
                    open=_to_float(values.get("1. open")),
                    high=_to_float(values.get("2. high")),
                    low=_to_float(values.get("3. low")),
                    adj_close=close,
                    volume=_to_int(values.get("5. volume")),
                    source=self.name,
                )
            )

        if not bars:
            raise DataNotFoundError(
                f"Alpha Vantage has no bars for '{symbol}' in {start}..{end}",
                provider=self.name,
            )
        bars.sort(key=lambda bar: bar.date)
        return bars

    def _request(self, params: dict[str, str], symbol: str) -> dict:
        """Spend one request from the budget and return the parsed payload."""
        if self._calls_made >= self._daily_budget:
            raise RateLimitError(
                f"Alpha Vantage budget of {self._daily_budget} requests is "
                "exhausted for this session",
                provider=self.name,
            )

        query = urllib.parse.urlencode({**params, "apikey": self._api_key})
        request = urllib.request.Request(
            f"{_ENDPOINT}?{query}", headers={"User-Agent": _USER_AGENT}
        )
        self._calls_made += 1
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise ProviderError(
                f"Alpha Vantage request failed for {symbol} (HTTP {exc.code})",
                provider=self.name,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                f"Could not reach Alpha Vantage for {symbol}: {exc}", provider=self.name
            ) from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Alpha Vantage returned invalid JSON for {symbol}: {exc}",
                provider=self.name,
            ) from exc

        # Alpha Vantage signals throttling and errors with HTTP 200 plus a
        # human-readable field, so the body has to be inspected explicitly.
        if "Note" in payload or "Information" in payload:
            message = payload.get("Note") or payload.get("Information", "")
            if "rate limit" in message.lower() or "requests per day" in message.lower():
                raise RateLimitError(
                    f"Alpha Vantage rate limit: {message}", provider=self.name
                )
            raise ProviderError(
                f"Alpha Vantage declined the request for {symbol}: {message}",
                provider=self.name,
            )
        if "Error Message" in payload:
            raise DataNotFoundError(
                f"Alpha Vantage rejected symbol '{symbol}': {payload['Error Message']}",
                provider=self.name,
            )
        return payload


def _is_plain_ticker(symbol: str) -> bool:
    """Whether this looks like a listed equity/ETF ticker Alpha Vantage accepts."""
    return symbol.isalpha() and 1 <= len(symbol) <= 5


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
