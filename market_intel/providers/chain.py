"""Fallback chain: try several market-data providers in priority order.

No free market-data source is simultaneously broad, current and officially
supported, so the recap does not bet on one:

* **yfinance** is the only source that is current across the whole universe, but
  it is an unofficial scrape that can break without notice.
* **FRED** is an official Federal Reserve service that will not change shape,
  but it publishes with a lag and maps to roughly thirty series.
* **Alpha Vantage** is a documented, licensed API - capped at 25 requests a day
  on the free tier, which is a fraction of one full refresh.

Chaining them means a single outage degrades the recap instead of emptying it.
Each :class:`~market_intel.models.PriceBar` already carries a ``source`` field,
so whichever provider actually served a number is recorded alongside it and the
UI can show provenance rather than implying one uniform feed.

Order matters and is deliberate: fastest-and-broadest first, most-authoritative
second, most-rate-limited last.
"""

from __future__ import annotations

import datetime as dt
import logging

from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)


class ChainedMarketDataProvider(MarketDataProvider):
    """Delegates to a list of providers, returning the first usable answer.

    A provider that raises :class:`DataNotFoundError` is treated as "does not
    cover this symbol" and the chain moves on quietly. Any other
    :class:`ProviderError` is treated as an outage: it is logged at warning level
    before moving on, because it means a source that *should* have answered
    did not.

    The chain only raises when every member has failed, and it re-raises the
    most informative error it saw - an outage if there was one, otherwise the
    "not found" from the last provider.
    """

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("A provider chain needs at least one provider")
        self._providers = providers
        self.name = "chain:" + "+".join(provider.name for provider in providers)

    @property
    def providers(self) -> list[MarketDataProvider]:
        """The underlying providers, in priority order."""
        return list(self._providers)

    def get_security_info(self, symbol: str) -> SecurityInfo:
        """Return the first provider's reference data for ``symbol``.

        Raises:
            DataNotFoundError: No provider in the chain knows this symbol.
            ProviderError: Every provider failed, at least one with an outage.
        """
        return self._first_success(
            "get_security_info", symbol, lambda provider: provider.get_security_info(symbol)
        )

    def get_daily_bars(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[PriceBar]:
        """Return daily bars from the first provider that supplies any.

        Raises:
            DataNotFoundError: No provider covers this symbol/range.
            ProviderError: Every provider failed, at least one with an outage.
        """

        def call(provider: MarketDataProvider) -> list[PriceBar]:
            bars = provider.get_daily_bars(symbol, start, end)
            if not bars:
                raise DataNotFoundError(
                    f"{provider.name} returned no bars for {symbol}",
                    provider=provider.name,
                )
            return bars

        return self._first_success("get_daily_bars", symbol, call)

    def _first_success(self, operation: str, symbol: str, call):
        """Run ``call`` against each provider until one succeeds."""
        outage: ProviderError | None = None
        not_found: DataNotFoundError | None = None

        for provider in self._providers:
            try:
                result = call(provider)
            except DataNotFoundError as exc:
                not_found = exc
                logger.debug(
                    "%s does not cover %s (%s)", provider.name, symbol, operation
                )
                continue
            except ProviderError as exc:
                outage = exc
                logger.warning(
                    "Provider %s failed on %s for %s: %s",
                    provider.name,
                    operation,
                    symbol,
                    exc,
                )
                continue
            except Exception as exc:  # a broken adapter must not break the chain
                outage = ProviderError(
                    f"{provider.name} raised an unexpected error: {exc}",
                    provider=provider.name,
                )
                logger.warning(
                    "Provider %s raised %s on %s for %s",
                    provider.name,
                    type(exc).__name__,
                    operation,
                    symbol,
                )
                continue

            if provider is not self._providers[0]:
                logger.info(
                    "Chain served %s for %s from fallback provider %s",
                    operation,
                    symbol,
                    provider.name,
                )
            return result

        if outage is not None:
            raise ProviderError(
                f"Every provider in {self.name} failed for '{symbol}': {outage}",
                provider=self.name,
            ) from outage
        raise DataNotFoundError(
            f"No provider in {self.name} covers '{symbol}'", provider=self.name
        ) from not_found
