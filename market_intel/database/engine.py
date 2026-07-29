"""Engine and session management.

The :class:`Database` wrapper is the single place that knows the
connection URL. Migrating to PostgreSQL means changing
``MIP_DATABASE_URL`` — no repository or service code changes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from market_intel.database.orm import Base

logger = logging.getLogger(__name__)


def _create_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with dialect-appropriate options."""
    url = make_url(database_url)
    kwargs: dict = {}

    if url.get_backend_name() == "sqlite":
        # Streamlit runs callbacks on worker threads.
        kwargs["connect_args"] = {"check_same_thread": False}
        if url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        else:
            # In-memory DBs need a single shared connection to be visible
            # across sessions (used by the test suite).
            kwargs["poolclass"] = StaticPool

    engine = create_engine(database_url, **kwargs)

    if url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


class Database:
    """Owns the engine and hands out transactional sessions."""

    def __init__(self, database_url: str) -> None:
        self.engine = _create_engine(database_url)
        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False
        )
        logger.info("Database engine created (%s)", self.engine.url.get_backend_name())

    def create_all(self) -> None:
        """Create any missing tables. Safe to call on every startup."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a session inside a commit/rollback transaction scope.

        Commits on clean exit, rolls back on any exception, always closes.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
