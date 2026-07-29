"""Calendar service: earnings and macro events."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories.events import EventRepository
from market_intel.exceptions import ProviderError
from market_intel.providers.base import EventProvider
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class CalendarService:
    """Maintains the event calendar: fetched earnings + manual macro events."""

    def __init__(
        self,
        db: Database,
        provider: EventProvider,
        cache: ApiCache,
        settings: Settings,
        market_data: MarketDataService,
    ) -> None:
        self._db = db
        self._provider = provider
        self._cache = cache
        self._settings = settings
        self._market_data = market_data

    def refresh_earnings(self, symbols: Sequence[str]) -> int:
        """Fetch and store earnings dates for each symbol (TTL-throttled).

        Returns:
            Number of events stored or updated.
        """
        stored = 0
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper()
            marker_key = f"earnings_synced:{self._provider.name}:{symbol}"
            if self._cache.get(marker_key) is not None:
                continue
            try:
                security_id = self._market_data.ensure_security(symbol)
                events = self._provider.get_earnings_events(symbol)
            except ProviderError as exc:
                logger.warning("Earnings refresh failed for %s: %s", symbol, exc)
                continue

            with self._db.session() as session:
                repo = EventRepository(session)
                for event in events:
                    repo.upsert(event, security_id=security_id)
            stored += len(events)
            self._cache.set(
                marker_key,
                {"refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                ttl_minutes=self._settings.calendar_cache_ttl_minutes,
            )
        logger.info("Earnings refresh stored %d events", stored)
        return stored

    def add_manual_event(
        self,
        title: str,
        scheduled_at: dt.datetime,
        event_type: str = "macro",
        notes: str | None = None,
    ) -> None:
        """Record a user-entered event (e.g. FOMC, CPI print)."""
        from market_intel.models import CalendarEvent

        event = CalendarEvent(
            event_type=event_type,
            title=title,
            scheduled_at=scheduled_at,
            notes=notes,
            source="manual",
        )
        with self._db.session() as session:
            EventRepository(session).upsert(event)

    def get_upcoming(
        self, days_ahead: int = 14, days_back: int = 0
    ) -> list[dict[str, Any]]:
        """Return events in the window as plain dicts, soonest first."""
        now = dt.datetime.now(dt.timezone.utc)
        start = now - dt.timedelta(days=days_back)
        end = now + dt.timedelta(days=days_ahead)
        with self._db.session() as session:
            events = EventRepository(session).get_between(start, end)
            return [
                {
                    "when": event.scheduled_at,
                    "type": event.event_type,
                    "title": event.title,
                    "symbol": event.security.symbol if event.security else None,
                    "period": event.period,
                    "consensus": event.consensus,
                    "actual": event.actual,
                    "notes": event.notes,
                }
                for event in events
            ]
