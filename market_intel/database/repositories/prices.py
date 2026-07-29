"""Repository for daily price history."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

from sqlalchemy import func, select, update

from market_intel.database import orm
from market_intel.database.orm import utcnow
from market_intel.database.repositories.base import BaseRepository
from market_intel.models import PriceBar

logger = logging.getLogger(__name__)


class PriceRepository(BaseRepository):
    """Idempotent storage and retrieval of daily OHLCV bars."""

    def upsert_bars(self, security_id: int, bars: Sequence[PriceBar]) -> int:
        """Insert new bars and refresh any that already exist for their date.

        Portable upsert (no dialect-specific ``ON CONFLICT``): existing
        (security, date) rows are updated, the rest bulk-inserted.

        Returns:
            Number of bars written.
        """
        if not bars:
            return 0

        dates = [bar.date for bar in bars]
        existing = set(
            self._session.execute(
                select(orm.PriceBar.date).where(
                    orm.PriceBar.security_id == security_id,
                    orm.PriceBar.date.in_(dates),
                )
            ).scalars()
        )

        for bar in bars:
            values = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "adj_close": bar.adj_close,
                "volume": bar.volume,
                "source": bar.source,
                "fetched_at": utcnow(),
            }
            if bar.date in existing:
                self._session.execute(
                    update(orm.PriceBar)
                    .where(
                        orm.PriceBar.security_id == security_id,
                        orm.PriceBar.date == bar.date,
                    )
                    .values(**values)
                )
            else:
                self._session.add(
                    orm.PriceBar(security_id=security_id, date=bar.date, **values)
                )

        self._session.flush()
        logger.debug(
            "Upserted %d bars for security_id=%d (%d updated)",
            len(bars),
            security_id,
            len(existing),
        )
        return len(bars)

    def get_history(
        self,
        security_id: int,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> list[orm.PriceBar]:
        """Return bars for a security ordered by date, optionally bounded."""
        stmt = (
            select(orm.PriceBar)
            .where(orm.PriceBar.security_id == security_id)
            .order_by(orm.PriceBar.date)
        )
        if start is not None:
            stmt = stmt.where(orm.PriceBar.date >= start)
        if end is not None:
            stmt = stmt.where(orm.PriceBar.date <= end)
        return list(self._session.execute(stmt).scalars())

    def latest_date(self, security_id: int) -> dt.date | None:
        """Most recent stored bar date, used for incremental fetches."""
        stmt = select(func.max(orm.PriceBar.date)).where(
            orm.PriceBar.security_id == security_id
        )
        return self._session.execute(stmt).scalar()

    def earliest_date(self, security_id: int) -> dt.date | None:
        """Oldest stored bar date, used to decide whether to backfill."""
        stmt = select(func.min(orm.PriceBar.date)).where(
            orm.PriceBar.security_id == security_id
        )
        return self._session.execute(stmt).scalar()
