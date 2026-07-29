"""Return and risk computations on price series."""

from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily returns from a close-price series (first row dropped)."""
    return close.pct_change().dropna()


def cumulative_return(close: pd.Series) -> float:
    """Total return over the whole series (0.10 = +10%)."""
    if len(close) < 2:
        return 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def annualised_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised standard deviation of a daily-returns series."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * (periods_per_year**0.5))


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """OLS beta of an asset to a benchmark on aligned daily returns.

    Returns None when there is not enough overlapping data or the
    benchmark has zero variance.
    """
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner")
    aligned.columns = ["asset", "bench"]
    aligned = aligned.dropna()
    if len(aligned) < 3:
        return None
    bench_var = aligned["bench"].var(ddof=1)
    if bench_var == 0:
        return None
    return float(aligned["asset"].cov(aligned["bench"]) / bench_var)
