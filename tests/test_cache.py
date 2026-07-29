"""Tests for the DB-backed TTL cache."""

import datetime as dt

import pytest

from market_intel.cache import ApiCache
from market_intel.database import Database
from market_intel.database.orm import ApiCacheEntry, utcnow


@pytest.fixture()
def cache() -> ApiCache:
    db = Database("sqlite:///:memory:")
    db.create_all()
    return ApiCache(db)


def test_roundtrip(cache: ApiCache) -> None:
    cache.set("key", {"a": 1, "b": [1, 2]}, ttl_minutes=5)
    assert cache.get("key") == {"a": 1, "b": [1, 2]}


def test_missing_key_returns_none(cache: ApiCache) -> None:
    assert cache.get("nope") is None


def test_overwrite_existing_key(cache: ApiCache) -> None:
    cache.set("key", "old", ttl_minutes=5)
    cache.set("key", "new", ttl_minutes=5)
    assert cache.get("key") == "new"


def test_expired_entry_returns_none_and_is_deleted(cache: ApiCache) -> None:
    cache.set("key", "value", ttl_minutes=5)
    # Force expiry by rewriting the entry with a past expires_at.
    with cache._db.session() as session:
        entry = session.get(ApiCacheEntry, "key")
        entry.expires_at = utcnow() - dt.timedelta(minutes=1)
        session.add(entry)

    assert cache.get("key") is None
    with cache._db.session() as session:
        assert session.get(ApiCacheEntry, "key") is None


def test_purge_expired(cache: ApiCache) -> None:
    cache.set("fresh", 1, ttl_minutes=60)
    cache.set("stale", 2, ttl_minutes=60)
    with cache._db.session() as session:
        entry = session.get(ApiCacheEntry, "stale")
        entry.expires_at = utcnow() - dt.timedelta(minutes=1)
        session.add(entry)

    assert cache.purge_expired() == 1
    assert cache.get("fresh") == 1
    assert cache.get("stale") is None
