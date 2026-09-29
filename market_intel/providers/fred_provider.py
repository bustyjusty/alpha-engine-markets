"""FRED adapter - official daily series from the St. Louis Fed.

Why this provider exists alongside yfinance:

* **Authority.** FRED is a published Federal Reserve service, not a scrape. Its
  Treasury-yield, credit-spread and breakeven series are the same numbers the
  sell side quotes, and the endpoint does not change shape without notice.
* **Coverage yfinance lacks.** ICE BofA option-adjusted spreads, the 10y-2y
  curve and 10-year breakeven inflation have no Yahoo ticker at all, but they
  are exactly what a macro desk reads first.
* **Free with no key.** The documented JSON API is used when ``MIP_FRED_API_KEY``
  is set; otherwise the adapter falls back to the public ``fredgraph.csv``
  endpoint, so the platform works out of the box.

**Known limitation - publication lag.** FRED posts daily series after the fact:
Treasury yields land the next business day and the H.10 FX rates are weekly.
FRED is therefore the *authoritative* source, not the *fastest* one, and belongs
behind a live feed in :class:`~market_intel.providers.chain.ChainedMarketDataProvider`
rather than in front of it.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from market_intel.exceptions import DataNotFoundError, ProviderError, RateLimitError
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_API_OBSERVATIONS = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT_SECONDS = 25
_USER_AGENT = "market-intel/0.1 (research dashboard)"


#: Universe symbol -> (FRED series id, display name, asset type).
#:
#: FX direction is checked against our quote convention: DEXJPUS is yen per
#: dollar (USD/JPY), DEXUSEU is dollars per euro (EUR/USD). Both already match
#: the universe, so no inversion is needed.
FRED_SERIES: dict[str, tuple[str, str, str]] = {
    # US Treasury yields (percent, same units as the Yahoo ^TNX family)
    "^IRX": ("DGS3MO", "US 3-month bill", "index"),
    "^FVX": ("DGS5", "US 5-year Treasury", "index"),
    "^TNX": ("DGS10", "US 10-year Treasury", "index"),
    "^TYX": ("DGS30", "US 30-year Treasury", "index"),
    # Equity indices
    "^GSPC": ("SP500", "S&P 500", "index"),
    "^NDX": ("NASDAQ100", "Nasdaq 100", "index"),
    "^DJI": ("DJIA", "Dow Jones Industrial Average", "index"),
    "^VIX": ("VIXCLS", "CBOE Volatility Index", "index"),
    # FX
    "EURUSD=X": ("DEXUSEU", "EUR/USD", "fx"),
    "GBPUSD=X": ("DEXUSUK", "GBP/USD", "fx"),
    "USDJPY=X": ("DEXJPUS", "USD/JPY", "fx"),
    "CNY=X": ("DEXCHUS", "USD/CNY", "fx"),
    "USDKRW=X": ("DEXKOUS", "USD/KRW", "fx"),
    "USDSGD=X": ("DEXSIUS", "USD/SGD", "fx"),
    "USDCHF=X": ("DEXSZUS", "USD/CHF", "fx"),
    "AUDUSD=X": ("DEXUSAL", "AUD/USD", "fx"),
    "USDCAD=X": ("DEXCAUS", "USD/CAD", "fx"),
    "USDINR=X": ("DEXINUS", "USD/INR", "fx"),
    # Commodities
    "CL=F": ("DCOILWTICO", "WTI crude", "commodity"),
    "BZ=F": ("DCOILBRENTEU", "Brent crude", "commodity"),
    "NG=F": ("DHHNGSP", "Henry Hub natural gas", "commodity"),
    # FRED-only macro series: no Yahoo equivalent exists for any of these.
    "FRED:HY_OAS": ("BAMLH0A0HYM2", "US high-yield OAS", "index"),
    "FRED:IG_OAS": ("BAMLC0A0CM", "US IG corporate OAS", "index"),
    "FRED:10Y2Y": ("T10Y2Y", "US 10y-2y curve", "index"),
    "FRED:BREAKEVEN10": ("T10YIE", "US 10-year breakeven inflation", "index"),
}


class FredProvider(MarketDataProvider):
    """Daily observations of FRED series, exposed as :class:`PriceBar` closes.

    FRED publishes a single value per day, so bars carry a ``close`` only;
    ``open``/``high``/``low``/``volume`` are left ``None`` rather than being
    faked from the close.
    """

    name = "fred"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or None

    def get_security_info(self, symbol: str) -> SecurityInfo:
        """Return reference data for a mapped symbol.

        Raises:
            DataNotFoundError: The symbol has no FRED series mapping.
        """
        entry = FRED_SERIES.get(symbol.strip().upper()) or FRED_SERIES.get(symbol.strip())
        if entry is None:
            raise DataNotFoundError(
                f"No FRED series mapped for '{symbol}'", provider=self.name
            )
        series_id, name, asset_type = entry
        return SecurityInfo(
            symbol=symbol.strip().upper(),
            name=name,
            asset_type=asset_type,
            currency="USD",
            exchange="FRED",
        )

    def get_daily_bars(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[PriceBar]:
        """Return daily observations for ``start``..``end`` inclusive.

        Raises:
            DataNotFoundError: The symbol is unmapped, or the window is empty.
            RateLimitError: FRED returned HTTP 429.
            ProviderError: Any other transport or parse failure.
        """
        key = symbol.strip().upper()
        entry = FRED_SERIES.get(key) or FRED_SERIES.get(symbol.strip())
        if entry is None:
            raise DataNotFoundError(
                f"No FRED series mapped for '{symbol}'", provider=self.name
            )
        series_id = entry[0]

        observations = (
            self._fetch_api(series_id, start, end)
            if self._api_key
            else self._fetch_csv(series_id, start, end)
        )

        bars = [
            PriceBar(date=day, close=value, adj_close=value, source=self.name)
            for day, value in observations
            if start <= day <= end
        ]
        if not bars:
            raise DataNotFoundError(
                f"FRED series {series_id} has no observations in "
                f"{start}..{end}",
                provider=self.name,
            )
        bars.sort(key=lambda bar: bar.date)
        return bars

    # --- Transport ------------------------------------------------------------

    def _fetch_csv(
        self, series_id: str, start: dt.date, end: dt.date
    ) -> list[tuple[dt.date, float]]:
        """Fetch via the keyless public CSV endpoint."""
        query = urllib.parse.urlencode(
            {"id": series_id, "cosd": start.isoformat(), "coed": end.isoformat()}
        )
        body = self._request(f"{_GRAPH_CSV}?{query}", series_id)

        rows: list[tuple[dt.date, float]] = []
        reader = csv.reader(io.StringIO(body))
        header = next(reader, None)
        if header is None:
            raise ProviderError(
                f"FRED returned an empty CSV for {series_id}", provider=self.name
            )
        for row in reader:
            if len(row) < 2:
                continue
            parsed = _parse_observation(row[0], row[1])
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _fetch_api(
        self, series_id: str, start: dt.date, end: dt.date
    ) -> list[tuple[dt.date, float]]:
        """Fetch via the documented JSON API (requires an API key)."""
        query = urllib.parse.urlencode(
            {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_start": start.isoformat(),
                "observation_end": end.isoformat(),
            }
        )
        body = self._request(f"{_API_OBSERVATIONS}?{query}", series_id)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"FRED returned invalid JSON for {series_id}: {exc}",
                provider=self.name,
            ) from exc

        rows: list[tuple[dt.date, float]] = []
        for observation in payload.get("observations", []):
            parsed = _parse_observation(
                observation.get("date", ""), observation.get("value", "")
            )
            if parsed is not None:
                rows.append(parsed)
        return rows

    def _request(self, url: str, series_id: str) -> str:
        """Perform one HTTP GET, translating transport errors to typed ones."""
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError(
                    f"FRED rate limit hit for {series_id}", provider=self.name
                ) from exc
            if exc.code == 404:
                raise DataNotFoundError(
                    f"FRED has no series {series_id}", provider=self.name
                ) from exc
            raise ProviderError(
                f"FRED request failed for {series_id} (HTTP {exc.code})",
                provider=self.name,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                f"Could not reach FRED for {series_id}: {exc}", provider=self.name
            ) from exc


def _parse_observation(raw_date: str, raw_value: str) -> tuple[dt.date, float] | None:
    """Parse one FRED row, skipping the '.' placeholders used for holidays."""
    raw_value = (raw_value or "").strip()
    if not raw_value or raw_value == ".":
        return None
    try:
        return dt.date.fromisoformat(raw_date.strip()), float(raw_value)
    except ValueError:
        return None
