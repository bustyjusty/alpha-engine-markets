"""Domain models for calendar events and ETF constituents."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """A dated market event (earnings release, macro print, custom note)."""

    event_type: str  # earnings | macro | custom
    title: str
    scheduled_at: dt.datetime
    symbol: str | None = None
    period: str | None = None
    consensus: float | None = None
    actual: float | None = None
    previous: float | None = None
    source: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class EtfHoldingItem:
    """One constituent of an ETF; ``weight`` is a fraction (0.07 = 7%)."""

    holding_symbol: str
    weight: float | None = None
    holding_name: str | None = None
