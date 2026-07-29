"""Repository for security reference data."""

from __future__ import annotations

import logging

from sqlalchemy import select

from market_intel.database import orm
from market_intel.database.repositories.base import BaseRepository
from market_intel.models import SecurityInfo

logger = logging.getLogger(__name__)

_MERGEABLE_FIELDS = ("name", "asset_type", "sector", "industry", "currency", "exchange")


class SecurityRepository(BaseRepository):
    """CRUD and upsert operations for :class:`~market_intel.database.orm.Security`."""

    def get_by_symbol(self, symbol: str) -> orm.Security | None:
        """Return the security with this symbol (case-insensitive), or None."""
        stmt = select(orm.Security).where(orm.Security.symbol == _normalise(symbol))
        return self._session.execute(stmt).scalar_one_or_none()

    def upsert(self, info: SecurityInfo) -> orm.Security:
        """Insert the security or merge non-null metadata into the existing row."""
        symbol = _normalise(info.symbol)
        security = self.get_by_symbol(symbol)
        if security is None:
            security = orm.Security(symbol=symbol)
            self._session.add(security)
            logger.debug("Inserting new security %s", symbol)

        for field in _MERGEABLE_FIELDS:
            value = getattr(info, field)
            if value is not None:
                setattr(security, field, value)

        self._session.flush()
        return security

    def list_active(self) -> list[orm.Security]:
        """Return all active securities ordered by symbol."""
        stmt = (
            select(orm.Security)
            .where(orm.Security.is_active.is_(True))
            .order_by(orm.Security.symbol)
        )
        return list(self._session.execute(stmt).scalars())


def _normalise(symbol: str) -> str:
    return symbol.strip().upper()
