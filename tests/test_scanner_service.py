"""Tests for ScannerService wiring (data flow, not the math)."""

import datetime as dt

import pytest

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.models import PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider
from market_intel.services import MarketDataService
from market_intel.services.scanner import ScannerService


class SyntheticProvider(MarketDataProvider):
    """Generates deterministic price paths per symbol."""

    name = "synthetic"

    # daily drift per symbol; SPIKE gets an extra late jump
    DRIFT = {"SPY": 1.001, "CALM": 1.001, "SPIKE": 1.001, "DOWN": 1.001}

    def get_security_info(self, symbol: str) -> SecurityInfo:
        if symbol not in self.DRIFT:
            raise DataNotFoundError(f"unknown {symbol}", provider=self.name)
        return SecurityInfo(symbol=symbol)

    def get_daily_bars(self, symbol, start, end) -> list[PriceBar]:
        if symbol not in self.DRIFT:
            raise ProviderError("unknown", provider=self.name)
        days = (end - start).days + 1
        bars = []
        price = 100.0
        for i in range(days):
            price *= self.DRIFT[symbol]
            if symbol != "SPY":
                price *= 1 + 0.002 * (-1) ** i  # noise so baselines have variance
            if symbol == "SPIKE" and i >= days - 5:
                price *= 1.02  # late 5-day melt-up
            if symbol == "DOWN" and i >= days - 5:
                price *= 0.98
            bars.append(
                PriceBar(date=start + dt.timedelta(days=i), close=price, source=self.name)
            )
        return bars


@pytest.fixture()
def scanner() -> ScannerService:
    db = Database("sqlite:///:memory:")
    db.create_all()
    settings = Settings(_env_file=None)
    market_data = MarketDataService(db, SyntheticProvider(), ApiCache(db), settings)
    return ScannerService(market_data)


def test_scan_ranks_movers_and_skips_bad_symbols(scanner: ScannerService) -> None:
    result = scanner.scan(
        ["SPIKE", "CALM", "DOWN", "BOGUS", "SPY"],  # SPY excluded (benchmark), BOGUS skipped
        benchmark_symbol="SPY",
        lookback_days=90,
    )

    assert set(result.index) == {"SPIKE", "CALM", "DOWN"}
    assert result.loc["SPIKE", "signal"] == "overreaction"
    assert result.loc["DOWN", "signal"] == "underreaction"
    assert result.loc["CALM", "signal"] == "normal"
    # Indicator enrichment is present on every row.
    for column in ("last_close", "rsi", "vwap_dev", "percent_b", "signals"):
        assert column in result.columns
    assert result.loc["SPIKE", "rsi"] is not None
    assert "overbought" in result.loc["SPIKE", "signals"]


def test_scan_with_unavailable_benchmark_returns_empty(scanner: ScannerService) -> None:
    result = scanner.scan(["CALM"], benchmark_symbol="BOGUS")
    assert result.empty
