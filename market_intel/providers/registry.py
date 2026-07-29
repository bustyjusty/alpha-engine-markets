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
}
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


def create_market_data_provider(settings: Settings) -> MarketDataProvider:
    """Instantiate the market-data provider selected by the settings.

    Raises:
        ConfigurationError: ``settings.market_data_provider`` is not registered.
    """
    return _instantiate(
        _MARKET_DATA_PROVIDERS, settings.market_data_provider.lower(), "market data"
    )


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
