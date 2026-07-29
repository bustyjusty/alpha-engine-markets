"""AI research service: generate and archive market research notes.

Uses the Anthropic API. The API key is optional — when absent the service
reports itself unavailable and the rest of the platform works normally.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories.research import ResearchRepository
from market_intel.exceptions import ConfigurationError, ProviderError, RateLimitError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior equity research analyst supporting discretionary, "
    "event-driven investing. Write objective, well-structured research notes "
    "in Markdown. Focus on: what happened, why it matters, which securities "
    "and sectors are affected, and what to monitor next. Clearly distinguish "
    "facts from interpretation, and note where data is missing or stale. "
    "This is decision-support research, not financial advice — do not present "
    "trade instructions."
)


class ResearchService:
    """Generates AI research notes and manages the research archive."""

    def __init__(
        self,
        db: Database,
        settings: Settings,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._client = client  # injectable for tests

    @property
    def available(self) -> bool:
        """Whether AI note generation can run (key configured or client injected)."""
        return self._client is not None or bool(self._settings.anthropic_api_key)

    def generate_note(
        self,
        title: str,
        context: str,
        focus: str | None = None,
        tags: str | None = None,
    ) -> dict[str, Any]:
        """Generate a research note from assembled market context and archive it.

        Args:
            title: Title for the archived note.
            context: The market data/news/events text the note should analyse.
            focus: Optional extra instruction (e.g. "focus on semiconductors").
            tags: Optional comma-separated tags for the archive.

        Raises:
            ConfigurationError: No API key configured.
            RateLimitError: The API rate limit was hit.
            ProviderError: Any other API failure, refusal, or empty response.
        """
        client = self._get_client()
        user_content = f"Prepare a research note based on the following market context.\n\n{context}"
        if focus:
            user_content += f"\n\nSpecific focus for this note: {focus}"

        try:
            response = client.messages.create(
                model=self._settings.ai_model,
                max_tokens=self._settings.ai_max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitError(
                f"Anthropic rate limit hit: {exc}", provider="anthropic"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error ({exc.status_code}): {exc.message}",
                provider="anthropic",
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Could not reach the Anthropic API: {exc}", provider="anthropic"
            ) from exc

        if response.stop_reason == "refusal":
            raise ProviderError(
                "The model declined to generate this note", provider="anthropic"
            )
        if response.stop_reason == "max_tokens":
            logger.warning("Research note hit the max_tokens limit and may be truncated")

        content = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not content:
            raise ProviderError("Model returned an empty note", provider="anthropic")

        with self._db.session() as session:
            note = ResearchRepository(session).add(
                title=title,
                content=content,
                note_type="ai_summary",
                model=response.model,
                tags=tags,
            )
            return _note_to_dict(note)

    def add_manual_note(
        self, title: str, content: str, tags: str | None = None
    ) -> dict[str, Any]:
        """Archive a user-written note."""
        with self._db.session() as session:
            note = ResearchRepository(session).add(
                title=title, content=content, note_type="manual", tags=tags
            )
            return _note_to_dict(note)

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._db.session() as session:
            return [_note_to_dict(n) for n in ResearchRepository(session).get_recent(limit)]

    def search(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._db.session() as session:
            return [_note_to_dict(n) for n in ResearchRepository(session).search(query, limit)]

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise ConfigurationError(
                    "MIP_ANTHROPIC_API_KEY is not set — AI research is disabled"
                )
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client


def compose_market_context(
    news: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    scan_summary: str | None = None,
) -> str:
    """Assemble service outputs into a text block for note generation."""
    sections: list[str] = []
    if news:
        lines = [
            f"- [{item.get('source') or 'unknown'}] {item['headline']}"
            + (f" ({', '.join(item['symbols'])})" if item.get("symbols") else "")
            for item in news
        ]
        sections.append("## Recent news\n" + "\n".join(lines))
    if events:
        lines = [
            f"- {event['when']:%Y-%m-%d}: {event['title']} [{event['type']}]"
            for event in events
        ]
        sections.append("## Upcoming events\n" + "\n".join(lines))
    if scan_summary:
        sections.append("## Relative-value scan\n" + scan_summary)
    return "\n\n".join(sections) if sections else "No market context available."


def _note_to_dict(note) -> dict[str, Any]:
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "note_type": note.note_type,
        "model": note.model,
        "tags": note.tags,
        "created_at": note.created_at,
    }
