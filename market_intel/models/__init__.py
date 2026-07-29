"""Provider-agnostic domain models.

These dataclasses are the contract between data providers, services and
repositories: providers return them, repositories persist them. Keeping
them independent of both the ORM and any provider payload format is what
makes API providers swappable.
"""

from market_intel.models.events import CalendarEvent, EtfHoldingItem
from market_intel.models.market import PriceBar, SecurityInfo
from market_intel.models.news import NewsItem

__all__ = [
    "CalendarEvent",
    "EtfHoldingItem",
    "NewsItem",
    "PriceBar",
    "SecurityInfo",
]
