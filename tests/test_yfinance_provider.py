"""Tests for the yfinance adapter.

The conversion logic is tested offline with a synthetic frame; a live
smoke test is marked ``integration`` and deselected by default
(run with: pytest -m integration).
"""

import datetime as dt
import math

import pandas as pd
import pytest

from market_intel.config import Settings
from market_intel.exceptions import ConfigurationError
from market_intel.providers.registry import create_market_data_provider
from market_intel.providers.yfinance_provider import frame_to_bars


def _history_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2026-07-06", "2026-07-07", "2026-07-08"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, math.nan],
            "High": [102.0, 103.0, math.nan],
            "Low": [99.0, 100.5, math.nan],
            "Close": [101.5, 102.5, math.nan],  # last row has no close
            "Adj Close": [101.5, 102.5, math.nan],
            "Volume": [1_000_000, math.nan, math.nan],
        },
        index=index,
    )


def test_frame_to_bars_conversion() -> None:
    bars = frame_to_bars(_history_frame(), source="yfinance")

    assert len(bars) == 2  # NaN-close row skipped
    first = bars[0]
    assert first.date == dt.date(2026, 7, 6)
    assert first.close == 101.5
    assert first.volume == 1_000_000
    assert first.source == "yfinance"
    assert bars[1].volume is None  # NaN volume -> None


def test_registry_creates_yfinance_provider() -> None:
    settings = Settings(_env_file=None)
    provider = create_market_data_provider(settings)
    assert provider.name == "yfinance"


def test_registry_rejects_unknown_provider() -> None:
    settings = Settings(_env_file=None, market_data_provider="bloomberg")
    with pytest.raises(ConfigurationError):
        create_market_data_provider(settings)


@pytest.mark.integration
def test_live_fetch_apple() -> None:
    """Live smoke test against Yahoo Finance."""
    provider = create_market_data_provider(Settings(_env_file=None))
    end = dt.date.today()
    bars = provider.get_daily_bars("AAPL", end - dt.timedelta(days=10), end)
    assert bars, "expected at least one bar in the last 10 days"
    assert all(b.close > 0 for b in bars)

    info = provider.get_security_info("AAPL")
    assert info.name and "Apple" in info.name
