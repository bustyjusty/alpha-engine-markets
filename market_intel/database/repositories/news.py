"""Repository for news articles and their security links."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from market_intel.database import orm
from market_intel.database.repositories.base import BaseRepository
from market_intel.models import NewsItem

logger = logging.getLogger(__name__)


class NewsRepository(BaseRepository):
    """Storage of news articles, de-duplicated by URL."""

    def upsert_article(
        self, item: NewsItem, security_ids: Sequence[int] = ()
    ) -> orm.NewsArticle:
        """Insert the article or enrich the existing row, linking securities.

        Existing security links are preserved; new ones are added, so an
        article fetched via several symbols accumulates all its links.
        """
        stmt = (
            select(orm.NewsArticle)
            .where(orm.NewsArticle.url == item.url)
            .options(selectinload(orm.NewsArticle.securities))
        )
        article = self._session.execute(stmt).scalar_one_or_none()
        if article is None:
            article = orm.NewsArticle(
                headline=item.headline,
                url=item.url,
                source=item.source,
                summary=item.summary,
                published_at=item.published_at,
            )
            self._session.add(article)

        linked_ids = {security.id for security in article.securities}
        for security_id in security_ids:
            if security_id not in linked_ids:
                security = self._session.get(orm.Security, security_id)
                if security is not None:
                    article.securities.append(security)

        self._session.flush()
        return article

    def get_recent(
        self, limit: int = 50, security_id: int | None = None
    ) -> list[orm.NewsArticle]:
        """Return the most recent articles, optionally for one security."""
        stmt = (
            select(orm.NewsArticle)
            .options(selectinload(orm.NewsArticle.securities))
            .order_by(orm.NewsArticle.published_at.desc())
            .limit(limit)
        )
        if security_id is not None:
            stmt = stmt.where(
                orm.NewsArticle.securities.any(orm.Security.id == security_id)
            )
        return list(self._session.execute(stmt).scalars())
