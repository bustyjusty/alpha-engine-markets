"""News service: collect, store and serve financial news."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories.news import NewsRepository
from market_intel.database.repositories.securities import SecurityRepository
from market_intel.exceptions import ProviderError
from market_intel.providers.base import NewsProvider
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class NewsService:
    """Collects news per symbol and serves the stored archive."""

    def __init__(
        self,
        db: Database,
        provider: NewsProvider,
        cache: ApiCache,
        settings: Settings,
        market_data: MarketDataService,
    ) -> None:
        self._db = db
        self._provider = provider
        self._cache = cache
        self._settings = settings
        self._market_data = market_data

    def refresh(self, symbols: Sequence[str], limit_per_symbol: int = 20) -> int:
        """Fetch and store recent news for each symbol.

        Each symbol is fetched at most once per ``news_cache_ttl_minutes``;
        provider failures are logged and skipped so one bad symbol never
        breaks a refresh sweep.

        Returns:
            Number of articles stored or updated.
        """
        stored = 0
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper()
            marker_key = f"news_synced:{self._provider.name}:{symbol}"
            if self._cache.get(marker_key) is not None:
                continue

            try:
                security_id = self._market_data.ensure_security(symbol)
                items = self._provider.get_news(symbol, limit=limit_per_symbol)
            except ProviderError as exc:
                logger.warning("News refresh failed for %s: %s", symbol, exc)
                continue

            with self._db.session() as session:
                repo = NewsRepository(session)
                for item in items:
                    repo.upsert_article(item, security_ids=[security_id])
            stored += len(items)

            self._cache.set(
                marker_key,
                {"refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                ttl_minutes=self._settings.news_cache_ttl_minutes,
            )
        logger.info("News refresh stored %d articles", stored)
        return stored

    def get_recent(
        self, limit: int = 50, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Return recent stored articles as plain dicts (newest first)."""
        with self._db.session() as session:
            security_id = None
            if symbol is not None:
                security = SecurityRepository(session).get_by_symbol(symbol)
                if security is None:
                    return []
                security_id = security.id
            articles = NewsRepository(session).get_recent(limit, security_id)
            return [
                {
                    "headline": article.headline,
                    "url": article.url,
                    "source": article.source,
                    "summary": article.summary,
                    "published_at": article.published_at,
                    "symbols": sorted(s.symbol for s in article.securities),
                }
                for article in articles
            ]
