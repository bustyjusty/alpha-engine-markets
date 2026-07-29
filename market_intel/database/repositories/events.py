"""Repository for calendar events."""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from market_intel.database import orm
from market_intel.database.repositories.base import BaseRepository
from market_intel.models import CalendarEvent

logger = logging.getLogger(__name__)


class EventRepository(BaseRepository):
    """Idempotent storage of earnings/macro/custom events."""

    def upsert(self, event: CalendarEvent, security_id: int | None = None) -> orm.Event:
        """Insert the event, or update an existing one on natural-key match.

        Natural key: (event_type, security_id, calendar day) for
        security-linked events, (event_type, title, calendar day) otherwise —
        so a re-fetched earnings date updates consensus/actual in place.
        """
        day_start = event.scheduled_at.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + dt.timedelta(days=1)
        stmt = select(orm.Event).where(
            orm.Event.event_type == event.event_type,
            orm.Event.scheduled_at >= day_start,
            orm.Event.scheduled_at < day_end,
        )
        if security_id is not None:
            stmt = stmt.where(orm.Event.security_id == security_id)
        else:
            stmt = stmt.where(orm.Event.title == event.title)
        existing = self._session.execute(stmt).scalars().first()

        if existing is None:
            existing = orm.Event(
                event_type=event.event_type,
                title=event.title,
                security_id=security_id,
                scheduled_at=event.scheduled_at,
            )
            self._session.add(existing)

        existing.scheduled_at = event.scheduled_at
        for field in ("period", "consensus", "actual", "previous", "source", "notes"):
            value = getattr(event, field)
            if value is not None:
                setattr(existing, field, value)

        self._session.flush()
        return existing

    def get_between(
        self,
        start: dt.datetime,
        end: dt.datetime,
        event_types: list[str] | None = None,
    ) -> list[orm.Event]:
        """Return events scheduled in [start, end), soonest first."""
        stmt = (
            select(orm.Event)
            .options(selectinload(orm.Event.security))
            .where(orm.Event.scheduled_at >= start, orm.Event.scheduled_at < end)
            .order_by(orm.Event.scheduled_at)
        )
        if event_types:
            stmt = stmt.where(orm.Event.event_type.in_(event_types))
        return list(self._session.execute(stmt).scalars())
