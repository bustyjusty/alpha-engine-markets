"""Adapters for external data APIs.

Each data category has an abstract interface in :mod:`.base`; concrete
adapters translate one provider's payloads into domain models and raise
the typed exceptions from :mod:`market_intel.exceptions`. Adapters are
instantiated through :mod:`.registry` based on settings.
"""

from market_intel.providers.base import MarketDataProvider, NewsProvider
from market_intel.providers.registry import (
    create_market_data_provider,
    create_news_provider,
)

__all__ = [
    "MarketDataProvider",
    "NewsProvider",
    "create_market_data_provider",
    "create_news_provider",
]
