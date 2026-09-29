"""Tests for the AI research service (Anthropic client mocked)."""

from types import SimpleNamespace

import pytest

from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import ConfigurationError, ProviderError
from market_intel.services.research import ResearchService, compose_market_context


class FakeMessages:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeAnthropicClient:
    def __init__(self, response=None, error=None):
        self.messages = FakeMessages(response, error)


def _response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="claude-opus-5",
    )


@pytest.fixture()
def db() -> Database:
    database = Database("sqlite:///:memory:")
    database.create_all()
    return database


def _service(db: Database, client) -> ResearchService:
    return ResearchService(db, Settings(_env_file=None), client=client)


class TestGenerateNote:
    def test_generates_and_archives(self, db: Database) -> None:
        client = FakeAnthropicClient(_response("## Market Note\nSemis rallied."))
        service = _service(db, client)

        note = service.generate_note(
            "Daily wrap", context="NVDA +5% on earnings", focus="semiconductors", tags="semis"
        )

        assert note["content"].startswith("## Market Note")
        assert note["model"] == "claude-opus-5"
        # Request carried the system prompt, model, and focus.
        call = client.messages.calls[0]
        assert call["model"] == "claude-opus-5"
        assert "event-driven" in call["system"]
        assert "semiconductors" in call["messages"][0]["content"]
        # Archived and retrievable.
        recent = service.get_recent()
        assert len(recent) == 1
        assert recent[0]["title"] == "Daily wrap"

    def test_refusal_raises_provider_error(self, db: Database) -> None:
        client = FakeAnthropicClient(_response("", stop_reason="refusal"))
        service = _service(db, client)
        with pytest.raises(ProviderError):
            service.generate_note("t", context="c")

    def test_no_api_key_raises_configuration_error(self, db: Database) -> None:
        service = ResearchService(db, Settings(_env_file=None))  # no client, no key
        assert not service.available
        with pytest.raises(ConfigurationError):
            service.generate_note("t", context="c")


class TestArchive:
    def test_manual_note_and_search(self, db: Database) -> None:
        service = _service(db, FakeAnthropicClient())
        service.add_manual_note("Fed watch", "FOMC likely holds in July.", tags="macro")
        service.add_manual_note("Chips", "TSMC capex up.")

        assert len(service.get_recent()) == 2
        hits = service.search("fomc")
        assert len(hits) == 1
        assert hits[0]["title"] == "Fed watch"
        assert service.search("nothing-matches") == []


def test_compose_market_context() -> None:
    import datetime as dt

    text = compose_market_context(
        news=[{"headline": "Chips rally", "source": "Reuters", "symbols": ["NVDA"]}],
        events=[{"when": dt.datetime(2026, 7, 15), "title": "NVDA earnings", "type": "earnings"}],
        scan_summary="NVDA z=+2.5 (overreaction)",
    )
    assert "Chips rally" in text
    assert "NVDA earnings" in text
    assert "z=+2.5" in text
    assert compose_market_context() == "No market context available."
