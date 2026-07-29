"""Pure analytical computations.

Everything in this package is side-effect free: functions take pandas
series/frames and return numbers or frames — no I/O, no database, no
providers — so it is trivially unit-testable and reusable.
"""

from market_intel.analysis.indicators import (
    bollinger,
    ema,
    indicator_snapshot,
    macd,
    rolling_twap,
    rolling_vwap,
    rsi,
    sma,
)
from market_intel.analysis.relative_value import ScanRow, relative_value_scan
from market_intel.analysis.returns import (
    annualised_volatility,
    beta,
    cumulative_return,
    daily_returns,
)

__all__ = [
    "ScanRow",
    "annualised_volatility",
    "beta",
    "bollinger",
    "cumulative_return",
    "daily_returns",
    "ema",
    "indicator_snapshot",
    "macd",
    "relative_value_scan",
    "rolling_twap",
    "rolling_vwap",
    "rsi",
    "sma",
]
