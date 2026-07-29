"""Tests for the technical indicators."""

import datetime as dt

import pandas as pd
import pytest

from market_intel.analysis import (
    bollinger,
    indicator_snapshot,
    rolling_twap,
    rolling_vwap,
    rsi,
    sma,
)


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range(dt.date(2026, 1, 1), periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def _frame(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "close": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
        },
        index=pd.date_range(dt.date(2026, 1, 1), periods=len(closes), freq="D"),
    )
    if volumes is not None:
        frame["volume"] = volumes
    return frame


class TestRsi:
    def test_pure_uptrend_saturates_at_100(self) -> None:
        values = rsi(_series([100 + i for i in range(30)]))
        assert values.iloc[-1] == pytest.approx(100.0)

    def test_pure_downtrend_approaches_zero(self) -> None:
        values = rsi(_series([100 - i for i in range(30)]))
        assert values.iloc[-1] < 5

    def test_nan_until_enough_history(self) -> None:
        values = rsi(_series([100, 101, 102]))
        assert values.isna().all()

    def test_balanced_moves_near_50(self) -> None:
        values = rsi(_series([100 + (1 if i % 2 else -1) for i in range(40)]))
        assert 35 < values.iloc[-1] < 65


class TestVwapTwap:
    def test_vwap_weights_by_volume(self) -> None:
        # Two prices, all the volume at the higher one -> VWAP near it.
        frame = _frame([100.0] * 10 + [110.0] * 10, volumes=[1] * 10 + [1_000_000] * 10)
        vwap = rolling_vwap(frame, window=20)
        twap = rolling_twap(frame, window=20)
        assert vwap.iloc[-1] == pytest.approx(110.0, rel=0.01)
        assert twap.iloc[-1] == pytest.approx(105.0, rel=0.01)  # equal-weighted

    def test_vwap_without_volume_is_nan(self) -> None:
        vwap = rolling_vwap(_frame([100.0] * 25), window=20)
        assert vwap.isna().all()


class TestBollinger:
    def test_constant_series_bands_collapse(self) -> None:
        upper, middle, lower = bollinger(_series([100.0] * 25))
        assert upper.iloc[-1] == middle.iloc[-1] == lower.iloc[-1] == 100.0

    def test_sma_matches_middle_band(self) -> None:
        closes = _series([100 + i * 0.5 for i in range(30)])
        _, middle, _ = bollinger(closes, window=20)
        assert middle.iloc[-1] == pytest.approx(sma(closes, 20).iloc[-1])


class TestSnapshot:
    def test_oversold_signal_on_persistent_selloff(self) -> None:
        frame = _frame([100 * (0.98**i) for i in range(40)], volumes=[1e6] * 40)
        snap = indicator_snapshot(frame)
        assert snap["rsi"] < 30
        assert "RSI oversold" in snap["signals"]
        assert "stretched below VWAP" in snap["signals"]
        assert snap["vwap_dev"] < 0

    def test_overbought_signal_on_melt_up(self) -> None:
        frame = _frame([100 * (1.02**i) for i in range(40)], volumes=[1e6] * 40)
        snap = indicator_snapshot(frame)
        assert snap["rsi"] > 70
        assert "RSI overbought" in snap["signals"]
        assert snap["vwap_dev"] > 0

    def test_quiet_tape_has_no_signals(self) -> None:
        closes = [100 + 0.1 * (1 if i % 2 else -1) for i in range(60)]
        snap = indicator_snapshot(_frame(closes, volumes=[1e6] * 60))
        assert snap["signals"] == []

    def test_short_history_returns_nones(self) -> None:
        snap = indicator_snapshot(_frame([100.0, 101.0]))
        assert snap["rsi"] is None
        assert snap["signals"] == []
        assert indicator_snapshot(pd.DataFrame())["close"] is None
