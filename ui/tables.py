"""Shared DataFrame styling — signed green/red numerics, terminal style.

Colors match the status pair used in ui.figures, brightened slightly for
small text on dark surfaces.
"""

from __future__ import annotations

import pandas as pd

POSITIVE = "#2fbf71"
NEGATIVE = "#f0616d"


def _signed_color(value: object) -> str:
    """CSS for one cell: green positive, red negative, default otherwise."""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return ""
    if value > 0:
        return f"color: {POSITIVE}; font-weight: 600;"
    if value < 0:
        return f"color: {NEGATIVE}; font-weight: 600;"
    return ""


def color_returns(frame: pd.DataFrame, columns: list[str]) -> "pd.io.formats.style.Styler":
    """Styler that colors the given numeric columns by sign.

    Columns missing from the frame are ignored, so callers can pass a
    superset. Number formats still come from st.column_config.
    """
    present = [column for column in columns if column in frame.columns]
    return frame.style.map(_signed_color, subset=present)
