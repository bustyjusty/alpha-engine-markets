"""Plotly figure builders — pure functions, no Streamlit calls.

Tuned for the dark theme. Colors follow the platform's viz rules:
status green/red only for gain/loss semantics (always paired with a
non-color cue like candle direction or bar sign), fixed categorical
hues for indicator overlays, and a blue↔red diverging pair for signed
values like z-scores.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Status (gain/loss semantics) — brightened for the black terminal surface
UP = "#2FD98A"
DOWN = "#FF5C6C"
# Categorical slots (dark-surface steps)
PRIMARY = "#5AA9FF"  # blue
VWAP_COLOR = "#C2F04A"  # terminal accent
SMA20_COLOR = "#38D9A9"  # aqua
SMA50_COLOR = "#B197FC"  # violet
NEGATIVE = "#FF6B81"  # diverging warm pole
# Chrome
MUTED = "#79828F"
GRID = "#1D232C"
BAND_FILL = "rgba(138, 147, 160, 0.12)"

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=36, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
)


def price_figure(
    frame: pd.DataFrame,
    symbol: str,
    overlays: dict[str, pd.Series] | None = None,
    bands: tuple[pd.Series, pd.Series] | None = None,
    rsi_series: pd.Series | None = None,
) -> go.Figure:
    """Candlestick + volume, with optional indicator overlays and RSI panel.

    Args:
        frame: OHLCV frame indexed by date.
        symbol: Title/legend name.
        overlays: Name -> series lines drawn on the price panel
            (VWAP / TWAP / SMAs); hues are assigned in fixed order.
        bands: (upper, lower) Bollinger bands drawn as a shaded channel.
        rsi_series: If given, adds an RSI panel with 30/70 guides.
    """
    overlays = overlays or {}
    has_rsi = rsi_series is not None
    rows = 3 if has_rsi else 2
    row_heights = [0.62, 0.18, 0.20] if has_rsi else [0.75, 0.25]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    if bands is not None:
        upper, lower = bands
        fig.add_trace(
            go.Scatter(
                x=frame.index, y=upper, line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index, y=lower, line=dict(width=0), fill="tonexty",
                fillcolor=BAND_FILL, name="Bollinger 20/2",
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"], high=frame["high"],
            low=frame["low"], close=frame["close"],
            increasing_line_color=UP, increasing_fillcolor=UP,
            decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
            name=symbol,
        ),
        row=1, col=1,
    )

    overlay_colors = {
        "VWAP": VWAP_COLOR, "TWAP": NEGATIVE,
        "SMA 20": SMA20_COLOR, "SMA 50": SMA50_COLOR,
    }
    for name, series in overlays.items():
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.values, mode="lines",
                line=dict(color=overlay_colors.get(name, PRIMARY), width=2),
                name=name,
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Bar(
            x=frame.index, y=frame["volume"],
            marker_color=PRIMARY, opacity=0.5, name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    if has_rsi:
        fig.add_trace(
            go.Scatter(
                x=rsi_series.index, y=rsi_series.values, mode="lines",
                line=dict(color=PRIMARY, width=2), name="RSI 14",
                showlegend=False,
            ),
            row=3, col=1,
        )
        for level in (30, 70):
            fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"), row=3, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", title_font=dict(size=11), row=3, col=1)

    fig.update_layout(
        **_BASE_LAYOUT,
        height=640 if has_rsi else 520,
        showlegend=bool(overlays or bands is not None),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{symbol} — daily", x=0),
    )
    for row in range(1, rows + 1):
        fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, row=row, col=1)
        fig.update_xaxes(showgrid=False, row=row, col=1)
    return fig


def line_figure(closes: pd.Series, symbol: str) -> go.Figure:
    """Single-series close line (no legend needed — the title names it)."""
    fig = go.Figure(
        go.Scatter(
            x=closes.index, y=closes.values, mode="lines",
            line=dict(color=PRIMARY, width=2), name=symbol,
        )
    )
    fig.update_layout(
        **_BASE_LAYOUT, height=300, showlegend=False, title=dict(text=symbol, x=0)
    )
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_xaxes(showgrid=False)
    return fig


def signed_bar_figure(
    labels: list[str], values: list[float], title: str, value_format: str = ".1%"
) -> go.Figure:
    """Horizontal bar chart of signed values: blue positive, red negative."""
    colors = [PRIMARY if value >= 0 else NEGATIVE for value in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors, marker_line_width=0,
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(240, 36 * len(labels) + 80),
        title=dict(text=title, x=0),
        xaxis=dict(tickformat=value_format, gridcolor=GRID, zerolinecolor=MUTED),
        yaxis=dict(autorange="reversed", showgrid=False),
        showlegend=False,
    )
    return fig
