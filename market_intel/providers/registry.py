"""Provider factory.

Adapter modules are imported lazily so that selecting one provider does
not require the dependencies of every other provider to be installed.
"""

from __future__ import annotations

import logging
from importlib import import_module

from market_intel.config import Settings
from market_intel.exceptions import ConfigurationError
from market_intel.providers.base import (
    EtfHoldingsProvider,
    EventProvider,
    MarketDataProvider,
    NewsProvider,
)

logger = logging.getLogger(__name__)

# name -> "module_path:ClassName"
_MARKET_DATA_PROVIDERS: dict[str, str] = {
    "yfinance": "market_intel.providers.yfinance_provider:YFinanceProvider",
    "fred": "market_intel.providers.fred_provider:FredProvider",
    "alphavantage": "market_intel.providers.alphavantage_provider:AlphaVantageProvider",
}

#: Priority order used by the ``chain`` market-data provider.
#:
#: Broadest and most current first, most authoritative second, most
#: rate-limited last - see :mod:`market_intel.providers.chain` for why no single
#: free source is sufficient on its own.
_CHAIN_ORDER: tuple[str, ...] = ("yfinance", "fred", "alphavantage")
_NEWS_PROVIDERS: dict[str, str] = {
    "yfinance": "market_intel.providers.yfinance_provider:YFinanceNewsProvider",
}
_EVENT_PROVIDERS: dict[str, str] = {
    "yfinance": "market_intel.providers.yfinance_provider:YFinanceEventProvider",
}
_ETF_PROVIDERS: dict[str, str] = {
    "yfinance": "market_intel.providers.yfinance_provider:YFinanceEtfProvider",
}


def _instantiate(registry: dict[str, str], name: str, kind: str):
    path = registry.get(name)
    if path is None:
        raise ConfigurationError(
            f"Unknown {kind} provider '{name}'. Registered: {sorted(registry)}"
        )
    module_path, _, class_name = path.partition(":")
    provider_cls = getattr(import_module(module_path), class_name)
    logger.info("Using %s provider: %s", kind, name)
    return provider_cls()


def _build_market_data_provider(name: str, settings: Settings) -> MarketDataProvider:
    """Construct one market-data adapter, passing whatever credentials it needs."""
    if name == "fred":
        from market_intel.providers.fred_provider import FredProvider

        return FredProvider(api_key=settings.fred_api_key)
    if name == "alphavantage":
        from market_intel.providers.alphavantage_provider import AlphaVantageProvider

        return AlphaVantageProvider(
            api_key=settings.alphavantage_api_key,
            daily_budget=settings.alphavantage_daily_budget,
        )
    return _instantiate(_MARKET_DATA_PROVIDERS, name, "market data")


def create_market_data_provider(settings: Settings) -> MarketDataProvider:
    """Instantiate the market-data provider selected by the settings.

    The special name ``chain`` builds a
    :class:`~market_intel.providers.chain.ChainedMarketDataProvider` over
    :data:`_CHAIN_ORDER`, so an outage in one free source degrades the data
    instead of emptying it. Adapters that fail to construct (usually a missing
    optional dependency) are dropped from the chain with a warning rather than
    taking the whole platform down.

    Raises:
        ConfigurationError: ``settings.market_data_provider`` is not registered,
            or every member of the chain failed to construct.
    """
    name = settings.market_data_provider.lower()
    if name != "chain":
        return _build_market_data_provider(name, settings)

    from market_intel.providers.chain import ChainedMarketDataProvider

    members: list[MarketDataProvider] = []
    for member_name in _CHAIN_ORDER:
        try:
            members.append(_build_market_data_provider(member_name, settings))
        except Exception as exc:
            logger.warning(
                "Provider '%s' unavailable, dropping it from the chain: %s",
                member_name,
                exc,
            )
    if not members:
        raise ConfigurationError(
            "No market-data provider in the chain could be constructed"
        )
    return ChainedMarketDataProvider(members)


def create_news_provider(settings: Settings) -> NewsProvider:
    """Instantiate the news provider selected by the settings.

    Raises:
        ConfigurationError: ``settings.news_provider`` is not registered.
    """
    return _instantiate(_NEWS_PROVIDERS, settings.news_provider.lower(), "news")


def create_event_provider(settings: Settings) -> EventProvider:
    """Instantiate the event provider selected by the settings."""
    return _instantiate(_EVENT_PROVIDERS, settings.event_provider.lower(), "event")


def create_etf_provider(settings: Settings) -> EtfHoldingsProvider:
    """Instantiate the ETF-holdings provider selected by the settings."""
    return _instantiate(_ETF_PROVIDERS, settings.etf_provider.lower(), "ETF holdings")
