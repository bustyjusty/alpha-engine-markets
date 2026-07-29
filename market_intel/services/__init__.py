"""Business-logic services.

Services orchestrate providers, repositories and the cache; they are the
only layer the UI talks to. Construction is wired in one place so the
dashboard and scripts share identical behaviour.
"""

from market_intel.services.market_data import MarketDataService
from market_intel.services.news import NewsService

__all__ = ["MarketDataService", "NewsService"]
