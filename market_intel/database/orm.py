"""SQLAlchemy ORM schema.

Portability rules (SQLite today, PostgreSQL tomorrow):
    * Only dialect-agnostic column types are used.
    * All datetimes are stored in UTC.
    * No dialect-specific features (e.g. SQLite ``ON CONFLICT`` upserts);
      upsert logic lives in the repositories using portable constructs.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    """Timezone-aware current UTC time, used for all audit columns."""
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class TimestampMixin:
    """Adds created/updated audit columns."""

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# --- Reference data ---------------------------------------------------------


class Security(TimestampMixin, Base):
    """A tradable security (equity, ETF, index, ...)."""

    __tablename__ = "securities"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    asset_type: Mapped[str] = mapped_column(String(20), default="equity")
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(100))
    currency: Mapped[str | None] = mapped_column(String(10))
    exchange: Mapped[str | None] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    price_bars: Mapped[list[PriceBar]] = relationship(
        back_populates="security", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Security {self.symbol}>"


# --- Market data ------------------------------------------------------------


class PriceBar(Base):
    """Daily OHLCV bar, with provenance (source, fetched_at)."""

    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("security_id", "date", name="uq_price_security_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    security_id: Mapped[int] = mapped_column(
        ForeignKey("securities.id", ondelete="CASCADE"), index=True
    )
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(50), default="unknown")
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    security: Mapped[Security] = relationship(back_populates="price_bars")


# --- News -------------------------------------------------------------------

article_securities = Table(
    "article_securities",
    Base.metadata,
    Column(
        "article_id",
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "security_id",
        ForeignKey("securities.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class NewsArticle(Base):
    """A collected news item, linkable to any number of securities."""

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    headline: Mapped[str] = mapped_column(String(512))
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(1024), unique=True)
    source: Mapped[str | None] = mapped_column(String(100))
    published_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    sentiment: Mapped[float | None] = mapped_column(Float)

    securities: Mapped[list[Security]] = relationship(secondary=article_securities)


# --- Events (earnings, macro) -----------------------------------------------


class Event(Base):
    """A dated market event: earnings release, macro print, or custom."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(30), index=True)  # earnings | macro | custom
    title: Mapped[str] = mapped_column(String(255))
    security_id: Mapped[int | None] = mapped_column(
        ForeignKey("securities.id", ondelete="SET NULL")
    )
    scheduled_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    period: Mapped[str | None] = mapped_column(String(20))  # e.g. "Q2 2026"
    consensus: Mapped[float | None] = mapped_column(Float)
    actual: Mapped[float | None] = mapped_column(Float)
    previous: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)

    security: Mapped[Security | None] = relationship()


# --- Themes & watchlists ------------------------------------------------------


class Theme(TimestampMixin, Base):
    """An investable theme (AI, semiconductors, defence, energy, ...)."""

    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str | None] = mapped_column(Text)

    members: Mapped[list[ThemeMember]] = relationship(
        back_populates="theme", cascade="all, delete-orphan"
    )


class ThemeMember(Base):
    """Association of a security to a theme, with optional weight/rationale."""

    __tablename__ = "theme_members"

    theme_id: Mapped[int] = mapped_column(
        ForeignKey("themes.id", ondelete="CASCADE"), primary_key=True
    )
    security_id: Mapped[int] = mapped_column(
        ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    weight: Mapped[float | None] = mapped_column(Float)
    note: Mapped[str | None] = mapped_column(String(255))

    theme: Mapped[Theme] = relationship(back_populates="members")
    security: Mapped[Security] = relationship()


class Watchlist(TimestampMixin, Base):
    """A named list of securities the user is monitoring."""

    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)

    members: Mapped[list[WatchlistMember]] = relationship(
        back_populates="watchlist", cascade="all, delete-orphan"
    )


class WatchlistMember(Base):
    """Association of a security to a watchlist."""

    __tablename__ = "watchlist_members"

    watchlist_id: Mapped[int] = mapped_column(
        ForeignKey("watchlists.id", ondelete="CASCADE"), primary_key=True
    )
    security_id: Mapped[int] = mapped_column(
        ForeignKey("securities.id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    note: Mapped[str | None] = mapped_column(String(255))

    watchlist: Mapped[Watchlist] = relationship(back_populates="members")
    security: Mapped[Security] = relationship()


# --- ETF constituents ---------------------------------------------------------


class EtfHolding(Base):
    """A single constituent of an ETF as of a given date.

    ``holding_symbol`` is stored as text (not a FK) because ETF holdings
    routinely include instruments we never track as securities (cash,
    futures, foreign lines).
    """

    __tablename__ = "etf_holdings"
    __table_args__ = (
        UniqueConstraint(
            "etf_security_id", "holding_symbol", "as_of", name="uq_etf_holding_asof"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    etf_security_id: Mapped[int] = mapped_column(
        ForeignKey("securities.id", ondelete="CASCADE"), index=True
    )
    holding_symbol: Mapped[str] = mapped_column(String(20))
    holding_name: Mapped[str | None] = mapped_column(String(255))
    weight: Mapped[float | None] = mapped_column(Float)  # fraction, e.g. 0.0712
    as_of: Mapped[dt.date] = mapped_column(Date)
    source: Mapped[str | None] = mapped_column(String(50))


# --- Research & journal -------------------------------------------------------


class ResearchNote(TimestampMixin, Base):
    """AI-generated or manual research note, archived for later review."""

    __tablename__ = "research_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    note_type: Mapped[str] = mapped_column(String(30), default="ai_summary")
    model: Mapped[str | None] = mapped_column(String(100))  # LLM used, if any
    tags: Mapped[str | None] = mapped_column(String(255))  # comma-separated


class JournalEntry(TimestampMixin, Base):
    """Trading-journal entry: idea, open position, or closed/reviewed trade."""

    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    security_id: Mapped[int | None] = mapped_column(
        ForeignKey("securities.id", ondelete="SET NULL")
    )
    direction: Mapped[str | None] = mapped_column(String(10))  # long | short
    status: Mapped[str] = mapped_column(String(20), default="idea")  # idea | open | closed
    thesis: Mapped[str] = mapped_column(Text)
    entry_price: Mapped[float | None] = mapped_column(Float)
    target_price: Mapped[float | None] = mapped_column(Float)
    stop_price: Mapped[float | None] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float)
    opened_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    review: Mapped[str | None] = mapped_column(Text)

    security: Mapped[Security | None] = relationship()


# --- API response cache -------------------------------------------------------


class ApiCacheEntry(Base):
    """TTL cache for expensive provider responses (JSON payloads)."""

    __tablename__ = "api_cache"

    cache_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    payload: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    expires_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
