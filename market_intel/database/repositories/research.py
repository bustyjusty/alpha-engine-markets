"""Repository for the research-note archive."""

from __future__ import annotations

from sqlalchemy import or_, select

from market_intel.database import orm
from market_intel.database.repositories.base import BaseRepository


class ResearchRepository(BaseRepository):
    """Storage and retrieval of AI-generated and manual research notes."""

    def add(
        self,
        title: str,
        content: str,
        note_type: str = "ai_summary",
        model: str | None = None,
        tags: str | None = None,
    ) -> orm.ResearchNote:
        note = orm.ResearchNote(
            title=title, content=content, note_type=note_type, model=model, tags=tags
        )
        self._session.add(note)
        self._session.flush()
        return note

    def get_recent(self, limit: int = 50) -> list[orm.ResearchNote]:
        stmt = (
            select(orm.ResearchNote)
            .order_by(orm.ResearchNote.created_at.desc(), orm.ResearchNote.id.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars())

    def search(self, query: str, limit: int = 50) -> list[orm.ResearchNote]:
        """Case-insensitive substring search over title, content and tags."""
        pattern = f"%{query}%"
        stmt = (
            select(orm.ResearchNote)
            .where(
                or_(
                    orm.ResearchNote.title.ilike(pattern),
                    orm.ResearchNote.content.ilike(pattern),
                    orm.ResearchNote.tags.ilike(pattern),
                )
            )
            .order_by(orm.ResearchNote.created_at.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars())
