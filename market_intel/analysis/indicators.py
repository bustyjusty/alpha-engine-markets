"""Technical indicators and signal snapshots.

Pure pandas computations on daily OHLCV frames (the shape returned by
``MarketDataService.get_price_history``). Notes for daily data:

* **VWAP/TWAP** are computed as *rolling* averages over N sessions using
  the typical price ((H+L+C)/3). Classic intraday VWAP needs tick data;
  the rolling variant is the standard daily approximation and answers
  the same question — is price stretched vs where volume traded?
* **RSI** uses Wilder's smoothing.

Everything degrades gracefully: missing volume disables VWAP, short
history yields ``None`` values instead of raising.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
VWAP_STRETCH = 0.02  # 2% from rolling VWAP counts as stretched


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder). 0–100; NaN until enough history."""
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # Pure uptrend: avg_loss == 0 -> rs inf -> RSI 100
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna(), other=float("nan"))


def typical_price(frame: pd.DataFrame) -> pd.Series:
    """(High + Low + Close) / 3, falling back to close where H/L is missing."""
    close = frame["close"].astype(float)
    if {"high", "low"}.issubset(frame.columns):
        high = frame["high"].astype(float).fillna(close)
        low = frame["low"].astype(float).fillna(close)
        return (high + low + close) / 3.0
    return close


def rolling_vwap(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling volume-weighted average price over ``window`` sessions.

    Returns an all-NaN series when volume is missing or zero throughout.
    """
    price = typical_price(frame)
    if "volume" not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    volume = frame["volume"].astype(float)
    weighted = (price * volume).rolling(window, min_periods=window).sum()
    total = volume.rolling(window, min_periods=window).sum()
    return weighted / total.where(total > 0)


def rolling_twap(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling time-weighted average price (equal-weight mean of typical price)."""
    return typical_price(frame).rolling(window, min_periods=window).mean()


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.astype(float).rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.astype(float).ewm(span=span, adjust=False).mean()


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands: (upper, middle, lower)."""
    middle = sma(close, window)
    std = close.astype(float).rolling(window, min_periods=window).std(ddof=0)
    return middle + num_std * std, middle, middle - num_std * std


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: (macd line, signal line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return line, signal_line, line - signal_line


def indicator_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    """Latest indicator values + human-readable signals for one symbol.

    Returns keys: close, rsi, vwap, vwap_dev, twap, twap_dev, sma20,
    sma50, percent_b, signals (list[str]). Values are None when there is
    not enough data to compute them.
    """
    if frame.empty or len(frame) < 2:
        return {"close": None, "rsi": None, "vwap": None, "vwap_dev": None,
                "twap": None, "twap_dev": None, "sma20": None, "sma50": None,
                "percent_b": None, "signals": []}

    closes = frame["close"].astype(float)
    last_close = float(closes.iloc[-1])

    rsi_value = _last(rsi(closes))
    vwap_value = _last(rolling_vwap(frame))
    twap_value = _last(rolling_twap(frame))
    sma20_value = _last(sma(closes, 20))
    sma50_value = _last(sma(closes, 50))
    upper, _, lower = bollinger(closes)
    upper_value, lower_value = _last(upper), _last(lower)
    macd_hist = macd(closes)[2]

    vwap_dev = last_close / vwap_value - 1.0 if vwap_value else None
    twap_dev = last_close / twap_value - 1.0 if twap_value else None
    percent_b = None
    if upper_value is not None and lower_value is not None and upper_value != lower_value:
        percent_b = (last_close - lower_value) / (upper_value - lower_value)

    signals: list[str] = []
    if rsi_value is not None:
        if rsi_value <= RSI_OVERSOLD:
            signals.append("RSI oversold")
        elif rsi_value >= RSI_OVERBOUGHT:
            signals.append("RSI overbought")
    if vwap_dev is not None:
        if vwap_dev >= VWAP_STRETCH:
            signals.append("stretched above VWAP")
        elif vwap_dev <= -VWAP_STRETCH:
            signals.append("stretched below VWAP")
    if percent_b is not None:
        if percent_b > 1.0:
            signals.append("above upper Bollinger band")
        elif percent_b < 0.0:
            signals.append("below lower Bollinger band")
    # MACD needs ~slow+signal sessions to mean anything, and micro-crosses
    # around zero on a quiet tape are noise — require a minimum magnitude.
    if len(closes) >= 35 and len(macd_hist.dropna()) >= 2:
        previous, current = float(macd_hist.iloc[-2]), float(macd_hist.iloc[-1])
        material = abs(current) > abs(last_close) * 1e-3
        if previous <= 0 < current and material:
            signals.append("MACD bullish cross")
        elif previous >= 0 > current and material:
            signals.append("MACD bearish cross")

    return {
        "close": last_close,
        "rsi": rsi_value,
        "vwap": vwap_value,
        "vwap_dev": vwap_dev,
        "twap": twap_value,
        "twap_dev": twap_dev,
        "sma20": sma20_value,
        "sma50": sma50_value,
        "percent_b": percent_b,
        "signals": signals,
    }


def _last(series: pd.Series) -> float | None:
    """Final value of a series as float, or None if NaN/empty."""
    if series.empty:
        return None
    value = series.iloc[-1]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)
