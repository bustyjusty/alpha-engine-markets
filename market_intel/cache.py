"""DB-backed TTL cache for expensive provider responses.

Backed by the ``api_cache`` table so entries survive process restarts and
Streamlit reruns, and shared caching helps respect provider rate limits.
Payloads must be JSON-serialisable.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

from sqlalchemy import delete

from market_intel.database import Database
from market_intel.database.orm import ApiCacheEntry, utcnow

logger = logging.getLogger(__name__)


class ApiCache:
    """Persistent key/value cache with per-entry expiry."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def get(self, key: str) -> Any | None:
        """Return the cached payload, or None if missing or expired.

        Expired entries are deleted on read.
        """
        with self._db.session() as session:
            entry = session.get(ApiCacheEntry, key)
            if entry is None:
                return None
            now = utcnow()
            if entry.expires_at.tzinfo is None:
                # SQLite round-trips datetimes as naive; values are always UTC.
                now = now.replace(tzinfo=None)
            if entry.expires_at <= now:
                session.delete(entry)
                logger.debug("Cache expired: %s", key)
                return None
            return json.loads(entry.payload)

    def set(self, key: str, value: Any, ttl_minutes: int) -> None:
        """Store a payload, replacing any existing entry for the key."""
        expires_at = utcnow() + dt.timedelta(minutes=ttl_minutes)
        with self._db.session() as session:
            session.merge(
                ApiCacheEntry(
                    cache_key=key,
                    payload=json.dumps(value),
                    created_at=utcnow(),
                    expires_at=expires_at,
                )
            )

    def purge_expired(self) -> int:
        """Delete all expired entries; returns how many were removed."""
        with self._db.session() as session:
            result = session.execute(
                delete(ApiCacheEntry).where(ApiCacheEntry.expires_at <= utcnow())
            )
            count = result.rowcount or 0
        if count:
            logger.info("Purged %d expired cache entries", count)
        return count
