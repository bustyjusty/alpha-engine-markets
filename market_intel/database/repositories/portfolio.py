"""Repositories for watchlists, themes and the trading journal."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from market_intel.database import orm
from market_intel.database.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class WatchlistRepository(BaseRepository):
    """CRUD for named watchlists and their members."""

    def get_or_create(self, name: str) -> orm.Watchlist:
        stmt = (
            select(orm.Watchlist)
            .where(orm.Watchlist.name == name)
            .options(selectinload(orm.Watchlist.members))
        )
        watchlist = self._session.execute(stmt).scalar_one_or_none()
        if watchlist is None:
            watchlist = orm.Watchlist(name=name)
            self._session.add(watchlist)
            self._session.flush()
        return watchlist

    def add_member(
        self, watchlist_id: int, security_id: int, note: str | None = None
    ) -> None:
        existing = self._session.get(orm.WatchlistMember, (watchlist_id, security_id))
        if existing is None:
            self._session.add(
                orm.WatchlistMember(
                    watchlist_id=watchlist_id, security_id=security_id, note=note
                )
            )
            self._session.flush()

    def remove_member(self, watchlist_id: int, security_id: int) -> bool:
        member = self._session.get(orm.WatchlistMember, (watchlist_id, security_id))
        if member is None:
            return False
        self._session.delete(member)
        self._session.flush()
        return True

    def list_all(self) -> list[orm.Watchlist]:
        stmt = select(orm.Watchlist).options(
            selectinload(orm.Watchlist.members).selectinload(
                orm.WatchlistMember.security
            )
        ).order_by(orm.Watchlist.name)
        return list(self._session.execute(stmt).scalars())

    def get_symbols(self, name: str) -> list[str]:
        stmt = (
            select(orm.Security.symbol)
            .join(orm.WatchlistMember, orm.WatchlistMember.security_id == orm.Security.id)
            .join(orm.Watchlist, orm.Watchlist.id == orm.WatchlistMember.watchlist_id)
            .where(orm.Watchlist.name == name)
            .order_by(orm.Security.symbol)
        )
        return list(self._session.execute(stmt).scalars())


class ThemeRepository(BaseRepository):
    """CRUD for investment themes and their members."""

    def get_or_create(self, name: str, description: str | None = None) -> orm.Theme:
        stmt = (
            select(orm.Theme)
            .where(orm.Theme.name == name)
            .options(selectinload(orm.Theme.members))
        )
        theme = self._session.execute(stmt).scalar_one_or_none()
        if theme is None:
            theme = orm.Theme(name=name, description=description)
            self._session.add(theme)
            self._session.flush()
        elif description is not None:
            theme.description = description
        return theme

    def add_member(
        self,
        theme_id: int,
        security_id: int,
        weight: float | None = None,
        note: str | None = None,
    ) -> None:
        member = self._session.get(orm.ThemeMember, (theme_id, security_id))
        if member is None:
            member = orm.ThemeMember(theme_id=theme_id, security_id=security_id)
            self._session.add(member)
        if weight is not None:
            member.weight = weight
        if note is not None:
            member.note = note
        self._session.flush()

    def remove_member(self, theme_id: int, security_id: int) -> bool:
        member = self._session.get(orm.ThemeMember, (theme_id, security_id))
        if member is None:
            return False
        self._session.delete(member)
        self._session.flush()
        return True

    def list_all(self) -> list[orm.Theme]:
        stmt = select(orm.Theme).options(
            selectinload(orm.Theme.members).selectinload(orm.ThemeMember.security)
        ).order_by(orm.Theme.name)
        return list(self._session.execute(stmt).scalars())

    def get_symbols(self, name: str) -> list[str]:
        stmt = (
            select(orm.Security.symbol)
            .join(orm.ThemeMember, orm.ThemeMember.security_id == orm.Security.id)
            .join(orm.Theme, orm.Theme.id == orm.ThemeMember.theme_id)
            .where(orm.Theme.name == name)
            .order_by(orm.Security.symbol)
        )
        return list(self._session.execute(stmt).scalars())


class JournalRepository(BaseRepository):
    """CRUD for trading-journal entries."""

    _UPDATABLE = {
        "direction",
        "status",
        "thesis",
        "entry_price",
        "target_price",
        "stop_price",
        "exit_price",
        "opened_at",
        "closed_at",
        "review",
        "security_id",
    }

    def add(self, thesis: str, **fields) -> orm.JournalEntry:
        unknown = set(fields) - self._UPDATABLE
        if unknown:
            raise ValueError(f"Unknown journal fields: {sorted(unknown)}")
        entry = orm.JournalEntry(thesis=thesis, **fields)
        self._session.add(entry)
        self._session.flush()
        return entry

    def update(self, entry_id: int, **fields) -> orm.JournalEntry | None:
        unknown = set(fields) - self._UPDATABLE
        if unknown:
            raise ValueError(f"Unknown journal fields: {sorted(unknown)}")
        entry = self._session.get(orm.JournalEntry, entry_id)
        if entry is None:
            return None
        for key, value in fields.items():
            setattr(entry, key, value)
        self._session.flush()
        return entry

    def list_entries(self, status: str | None = None) -> list[orm.JournalEntry]:
        stmt = (
            select(orm.JournalEntry)
            .options(selectinload(orm.JournalEntry.security))
            .order_by(orm.JournalEntry.created_at.desc(), orm.JournalEntry.id.desc())
        )
        if status is not None:
            stmt = stmt.where(orm.JournalEntry.status == status)
        return list(self._session.execute(stmt).scalars())
