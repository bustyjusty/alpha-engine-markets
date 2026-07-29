"""Repositories: the only layer that touches the ORM directly.

Services depend on repositories, never on SQLAlchemy queries, so storage
concerns stay in one place. Further repositories (news, events, themes,
journal) are added alongside the modules that need them.
"""

from market_intel.database.repositories.prices import PriceRepository
from market_intel.database.repositories.securities import SecurityRepository

__all__ = ["PriceRepository", "SecurityRepository"]
