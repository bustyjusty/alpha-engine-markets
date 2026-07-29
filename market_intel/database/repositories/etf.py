"""Repository for ETF constituent snapshots."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

from sqlalchemy import delete, select

from market_intel.database import orm
from market_intel.database.repositories.base import BaseRepository
from market_intel.models import EtfHoldingItem

logger = logging.getLogger(__name__)


class EtfHoldingRepository(BaseRepository):
    """Point-in-time snapshots of ETF holdings, keyed by (etf, as_of)."""

    def replace_snapshot(
        self,
        etf_security_id: int,
        items: Sequence[EtfHoldingItem],
        as_of: dt.date,
        source: str | None = None,
    ) -> int:
        """Replace the holdings snapshot for an ETF on a given date."""
        self._session.execute(
            delete(orm.EtfHolding).where(
                orm.EtfHolding.etf_security_id == etf_security_id,
                orm.EtfHolding.as_of == as_of,
            )
        )
        for item in items:
            self._session.add(
                orm.EtfHolding(
                    etf_security_id=etf_security_id,
                    holding_symbol=item.holding_symbol,
                    holding_name=item.holding_name,
                    weight=item.weight,
                    as_of=as_of,
                    source=source,
                )
            )
        self._session.flush()
        return len(items)

    def latest_snapshot(self, etf_security_id: int) -> list[orm.EtfHolding]:
        """Return the most recent snapshot for an ETF, largest weight first."""
        latest_stmt = select(orm.EtfHolding.as_of).where(
            orm.EtfHolding.etf_security_id == etf_security_id
        ).order_by(orm.EtfHolding.as_of.desc()).limit(1)
        latest = self._session.execute(latest_stmt).scalar()
        if latest is None:
            return []
        stmt = (
            select(orm.EtfHolding)
            .where(
                orm.EtfHolding.etf_security_id == etf_security_id,
                orm.EtfHolding.as_of == latest,
            )
            .order_by(orm.EtfHolding.weight.desc())
        )
        return list(self._session.execute(stmt).scalars())

    def exposure_for_symbol(self, holding_symbol: str) -> list[tuple[orm.Security, orm.EtfHolding]]:
        """Return (ETF security, holding row) pairs for every tracked ETF whose
        latest snapshot contains the symbol."""
        stmt = (
            select(orm.Security, orm.EtfHolding)
            .join(orm.EtfHolding, orm.EtfHolding.etf_security_id == orm.Security.id)
            .where(orm.EtfHolding.holding_symbol == holding_symbol.strip().upper())
            .order_by(orm.EtfHolding.as_of.desc())
        )
        rows = list(self._session.execute(stmt))
        # Keep only the row from each ETF's latest snapshot.
        result: list[tuple[orm.Security, orm.EtfHolding]] = []
        seen: set[int] = set()
        for security, holding in rows:
            if security.id in seen:
                continue
            latest = self.latest_snapshot(security.id)
            if any(h.id == holding.id for h in latest):
                result.append((security, holding))
            seen.add(security.id)
        return result
