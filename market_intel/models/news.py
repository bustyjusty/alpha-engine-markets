"""Domain models for news."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NewsItem:
    """A provider-agnostic news article.

    ``url`` is the de-duplication key: the same story fetched for two
    symbols is stored once and linked to both securities.
    """

    headline: str
    url: str
    source: str | None = None
    summary: str | None = None
    published_at: dt.datetime | None = None
    symbols: tuple[str, ...] = ()
