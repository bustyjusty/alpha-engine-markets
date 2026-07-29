"""Tests for the pure analysis functions and the relative-value scan."""

import datetime as dt

import pandas as pd
import pytest

from market_intel.analysis import (
    annualised_volatility,
    beta,
    cumulative_return,
    daily_returns,
    relative_value_scan,
)


def _series(values: list[float], start: dt.date = dt.date(2026, 1, 1)) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


class TestReturns:
    def test_daily_returns(self) -> None:
        returns = daily_returns(_series([100, 110, 99]))
        assert returns.iloc[0] == pytest.approx(0.10)
        assert returns.iloc[1] == pytest.approx(-0.10)

    def test_cumulative_return(self) -> None:
        assert cumulative_return(_series([100, 90, 120])) == pytest.approx(0.20)
        assert cumulative_return(_series([100])) == 0.0

    def test_annualised_volatility_of_constant_returns_is_zero(self) -> None:
        constant = daily_returns(_series([100, 101, 102.01]))
        assert annualised_volatility(constant) == pytest.approx(0.0, abs=1e-9)

    def test_beta_of_scaled_series_is_the_scale(self) -> None:
        bench = daily_returns(_series([100, 102, 101, 104, 103, 106]))
        asset = bench * 1.5
        assert beta(asset, bench) == pytest.approx(1.5)

    def test_beta_insufficient_data_returns_none(self) -> None:
        assert beta(_series([0.01]), _series([0.01])) is None


def _wiggle(series: pd.Series, amp: float = 0.002) -> pd.Series:
    """Deterministic alternating noise so relative returns have nonzero std."""
    factors = [1 + amp * (-1) ** i for i in range(len(series))]
    return series * factors


class TestRelativeValueScan:
    def _flat_benchmark(self, days: int = 80) -> pd.Series:
        # Benchmark drifting up steadily.
        return _series([100 * (1.001**i) for i in range(days)])

    def test_spike_is_flagged_as_overreaction(self) -> None:
        bench = self._flat_benchmark()
        # Roughly tracks the benchmark, then jumps ~10% over the final 5 days.
        spiky = _wiggle(bench)
        values = list(spiky.values)
        for i in range(len(values) - 5, len(values)):
            values[i] = values[i] * (1.02 ** (i - (len(values) - 6)))
        spiky = pd.Series(values, index=bench.index)

        result = relative_value_scan(
            {"SPIKE": spiky, "CALM": _wiggle(bench * 0.5)}, bench
        )

        assert result.loc["SPIKE", "signal"] == "overreaction"
        assert result.loc["SPIKE", "zscore"] > 2
        assert result.loc["CALM", "signal"] == "normal"
        # Ranked by |z|: SPIKE first.
        assert list(result.index)[0] == "SPIKE"

    def test_drop_is_flagged_as_underreaction(self) -> None:
        bench = self._flat_benchmark()
        values = list(_wiggle(bench).values)
        for i in range(len(values) - 5, len(values)):
            values[i] = values[i] * (0.98 ** (i - (len(values) - 6)))
        dippy = pd.Series(values, index=bench.index)

        result = relative_value_scan({"DIP": dippy}, bench)
        assert result.loc["DIP", "signal"] == "underreaction"
        assert result.loc["DIP", "zscore"] < -2

    def test_insufficient_history(self) -> None:
        bench = self._flat_benchmark()
        short = _series([100, 101, 102])
        result = relative_value_scan({"NEW": short}, bench)
        assert result.loc["NEW", "signal"] == "insufficient_data"
        assert result.loc["NEW", "zscore"] is None or pd.isna(
            result.loc["NEW", "zscore"]
        )

    def test_relative_return_is_symbol_minus_benchmark(self) -> None:
        bench = self._flat_benchmark()
        result = relative_value_scan({"SAME": bench.copy()}, bench)
        assert result.loc["SAME", "relative_return"] == pytest.approx(0.0, abs=1e-12)
