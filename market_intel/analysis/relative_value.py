"""Relative-value scan: which names moved unusually vs their benchmark.

Method:
    1. Align each symbol's closes with the benchmark's (inner join).
    2. Compute the rolling ``window_days`` relative return
       (symbol window return minus benchmark window return) through time.
    3. Z-score the latest window against the symbol's own history of
       windows.

A large positive z-score means the name just outperformed its benchmark
by far more than is normal for it (possible over-reaction / rich); a
large negative z-score flags a potential under-reaction / cheap name.
The output ranks candidates for discretionary review — it is not a
trading signal.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScanRow:
    """One symbol's result in a relative-value scan."""

    symbol: str
    window_return: float
    benchmark_return: float
    relative_return: float
    zscore: float | None
    signal: str  # overreaction | underreaction | normal | insufficient_data


def relative_value_scan(
    prices: dict[str, pd.Series],
    benchmark: pd.Series,
    window_days: int = 5,
    min_history: int = 40,
    zscore_threshold: float = 2.0,
) -> pd.DataFrame:
    """Scan symbols for unusual moves relative to a benchmark.

    Args:
        prices: Symbol -> date-indexed close series.
        benchmark: Date-indexed close series of the benchmark.
        window_days: Length of the reaction window being scored.
        min_history: Minimum aligned observations required to score.
        zscore_threshold: |z| beyond which a name is flagged.

    Returns:
        DataFrame indexed by symbol with return/z-score columns, sorted
        by absolute z-score descending (unscoreable names last).
    """
    rows: list[ScanRow] = []
    for symbol, close in prices.items():
        rows.append(
            _scan_one(
                symbol, close, benchmark, window_days, min_history, zscore_threshold
            )
        )

    frame = pd.DataFrame([asdict(row) for row in rows]).set_index("symbol")
    frame["zscore"] = pd.to_numeric(frame["zscore"], errors="coerce")
    frame["abs_z"] = frame["zscore"].abs()
    frame = frame.sort_values("abs_z", ascending=False, na_position="last")
    return frame.drop(columns="abs_z")


def _scan_one(
    symbol: str,
    close: pd.Series,
    benchmark: pd.Series,
    window_days: int,
    min_history: int,
    zscore_threshold: float,
) -> ScanRow:
    aligned = pd.concat([close, benchmark], axis=1, join="inner").dropna()
    aligned.columns = ["sym", "bench"]

    if len(aligned) < max(min_history, window_days + 2):
        logger.debug("Insufficient history for %s (%d rows)", symbol, len(aligned))
        return ScanRow(symbol, 0.0, 0.0, 0.0, None, "insufficient_data")

    sym_window = aligned["sym"] / aligned["sym"].shift(window_days) - 1.0
    bench_window = aligned["bench"] / aligned["bench"].shift(window_days) - 1.0
    relative = (sym_window - bench_window).dropna()

    current = float(relative.iloc[-1])
    # Baseline excludes windows overlapping the current one.
    baseline = relative.iloc[: -window_days if window_days < len(relative) else -1]
    zscore: float | None = None
    if len(baseline) >= 10:
        std = float(baseline.std(ddof=1))
        if std > 0:
            zscore = (current - float(baseline.mean())) / std

    if zscore is None:
        signal = "insufficient_data"
    elif zscore >= zscore_threshold:
        signal = "overreaction"
    elif zscore <= -zscore_threshold:
        signal = "underreaction"
    else:
        signal = "normal"

    return ScanRow(
        symbol=symbol,
        window_return=float(sym_window.iloc[-1]),
        benchmark_return=float(bench_window.iloc[-1]),
        relative_return=current,
        zscore=zscore,
        signal=signal,
    )
