"""Shared repository plumbing."""

from __future__ import annotations

from sqlalchemy.orm import Session


class BaseRepository:
    """Base class binding a repository to an active session.

    Repositories are cheap, short-lived objects created inside a
    ``Database.session()`` scope; they never own or commit the session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
