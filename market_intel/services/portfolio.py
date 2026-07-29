"""Watchlist, theme and journal services.

Thin orchestration over the portfolio repositories: symbols in, plain
dicts out, with security records resolved via MarketDataService.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from market_intel.database import Database
from market_intel.database.repositories.portfolio import (
    JournalRepository,
    ThemeRepository,
    WatchlistRepository,
)
from market_intel.exceptions import ProviderError
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class WatchlistService:
    """Manage named lists of symbols to monitor."""

    def __init__(self, db: Database, market_data: MarketDataService) -> None:
        self._db = db
        self._market_data = market_data

    def add_symbol(self, watchlist_name: str, symbol: str, note: str | None = None) -> None:
        security_id = self._market_data.ensure_security(symbol)
        with self._db.session() as session:
            repo = WatchlistRepository(session)
            watchlist = repo.get_or_create(watchlist_name)
            repo.add_member(watchlist.id, security_id, note=note)

    def remove_symbol(self, watchlist_name: str, symbol: str) -> bool:
        with self._db.session() as session:
            repo = WatchlistRepository(session)
            watchlist = repo.get_or_create(watchlist_name)
            from market_intel.database.repositories.securities import SecurityRepository

            security = SecurityRepository(session).get_by_symbol(symbol)
            if security is None:
                return False
            return repo.remove_member(watchlist.id, security.id)

    def get_symbols(self, watchlist_name: str) -> list[str]:
        with self._db.session() as session:
            return WatchlistRepository(session).get_symbols(watchlist_name)

    def list_watchlists(self) -> list[dict[str, Any]]:
        with self._db.session() as session:
            return [
                {
                    "name": wl.name,
                    "symbols": sorted(m.security.symbol for m in wl.members),
                }
                for wl in WatchlistRepository(session).list_all()
            ]


class ThemeService:
    """Manage investment themes and report their recent performance."""

    def __init__(self, db: Database, market_data: MarketDataService) -> None:
        self._db = db
        self._market_data = market_data

    def create_theme(self, name: str, description: str | None = None) -> None:
        with self._db.session() as session:
            ThemeRepository(session).get_or_create(name, description)

    def add_member(
        self,
        theme_name: str,
        symbol: str,
        weight: float | None = None,
        note: str | None = None,
    ) -> None:
        security_id = self._market_data.ensure_security(symbol)
        with self._db.session() as session:
            repo = ThemeRepository(session)
            theme = repo.get_or_create(theme_name)
            repo.add_member(theme.id, security_id, weight=weight, note=note)

    def list_themes(self) -> list[dict[str, Any]]:
        with self._db.session() as session:
            return [
                {
                    "name": theme.name,
                    "description": theme.description,
                    "members": [
                        {
                            "symbol": m.security.symbol,
                            "weight": m.weight,
                            "note": m.note,
                        }
                        for m in theme.members
                    ],
                }
                for theme in ThemeRepository(session).list_all()
            ]

    def get_symbols(self, theme_name: str) -> list[str]:
        with self._db.session() as session:
            return ThemeRepository(session).get_symbols(theme_name)

    def performance(self, theme_name: str, days: int = 30) -> list[dict[str, Any]]:
        """Per-member return over the window; unfetchable symbols are skipped."""
        start = dt.date.today() - dt.timedelta(days=days)
        rows: list[dict[str, Any]] = []
        for symbol in self.get_symbols(theme_name):
            try:
                frame = self._market_data.get_price_history(symbol, start=start)
            except ProviderError as exc:
                logger.warning("Skipping %s in theme performance: %s", symbol, exc)
                continue
            if len(frame) < 2:
                continue
            closes = frame["adj_close"].fillna(frame["close"]).astype(float)
            rows.append(
                {
                    "symbol": symbol,
                    "return": float(closes.iloc[-1] / closes.iloc[0] - 1.0),
                    "last_close": float(closes.iloc[-1]),
                }
            )
        rows.sort(key=lambda row: row["return"], reverse=True)
        return rows


class JournalService:
    """Trading journal: ideas, open positions, closed trades with review."""

    def __init__(self, db: Database, market_data: MarketDataService) -> None:
        self._db = db
        self._market_data = market_data

    def log_idea(
        self,
        thesis: str,
        symbol: str | None = None,
        direction: str | None = None,
        entry_price: float | None = None,
        target_price: float | None = None,
        stop_price: float | None = None,
    ) -> int:
        security_id = self._market_data.ensure_security(symbol) if symbol else None
        with self._db.session() as session:
            entry = JournalRepository(session).add(
                thesis=thesis,
                security_id=security_id,
                direction=direction,
                status="idea",
                entry_price=entry_price,
                target_price=target_price,
                stop_price=stop_price,
            )
            return entry.id

    def open_position(self, entry_id: int, entry_price: float) -> bool:
        with self._db.session() as session:
            entry = JournalRepository(session).update(
                entry_id,
                status="open",
                entry_price=entry_price,
                opened_at=dt.datetime.now(dt.timezone.utc),
            )
            return entry is not None

    def close_position(
        self, entry_id: int, exit_price: float, review: str | None = None
    ) -> bool:
        with self._db.session() as session:
            entry = JournalRepository(session).update(
                entry_id,
                status="closed",
                exit_price=exit_price,
                closed_at=dt.datetime.now(dt.timezone.utc),
                review=review,
            )
            return entry is not None

    def list_entries(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._db.session() as session:
            entries = JournalRepository(session).list_entries(status)
            return [
                {
                    "id": entry.id,
                    "symbol": entry.security.symbol if entry.security else None,
                    "direction": entry.direction,
                    "status": entry.status,
                    "thesis": entry.thesis,
                    "entry_price": entry.entry_price,
                    "target_price": entry.target_price,
                    "stop_price": entry.stop_price,
                    "exit_price": entry.exit_price,
                    "pnl_pct": _pnl_pct(entry),
                    "opened_at": entry.opened_at,
                    "closed_at": entry.closed_at,
                    "review": entry.review,
                    "created_at": entry.created_at,
                }
                for entry in entries
            ]


def _pnl_pct(entry) -> float | None:
    """Realised P&L percentage for a closed trade (sign-aware for shorts)."""
    if entry.entry_price in (None, 0) or entry.exit_price is None:
        return None
    raw = entry.exit_price / entry.entry_price - 1.0
    return -raw if entry.direction == "short" else raw
