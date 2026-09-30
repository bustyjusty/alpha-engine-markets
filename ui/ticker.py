"""The loaded security — one ticker, shared by every single-name page.

Bloomberg and Capital IQ are security-first: you load a name once and every
view hangs off it. This dashboard used to be page-first — eleven destinations,
each with its own ticker box, so moving from a chart to an analysis meant
retyping the symbol and hoping both had loaded the same thing.

The loaded symbol lives in session state, the bar below renders it, and every
page under the SECURITY section reads it instead of asking again.
"""

from __future__ import annotations

import streamlit as st

from market_intel.exceptions import MarketIntelError
from market_intel.formatting import money_formatter

#: Where the loaded symbol lives. Distinct from the input widget's own key —
#: Streamlit forbids writing to a widget's key after the widget is created.
_SYMBOL_KEY = "ae_symbol"
_INPUT_KEY = "ae_symbol_input"
_RECENT_KEY = "ae_recent"

#: Loaded on a cold session. A liquid, always-quoted name, so the first render
#: of any page shows real data rather than an error.
DEFAULT_SYMBOL = "SPY"
#: How many recently loaded tickers stay clickable in the bar.
MAX_RECENT = 6
#: Trailing sessions drawn in the header sparkline.
SPARK_SESSIONS = 90


def loaded_symbol() -> str:
    """The ticker every security page is currently showing."""
    return st.session_state.get(_SYMBOL_KEY, DEFAULT_SYMBOL)


def set_symbol(symbol: str) -> None:
    """Load a ticker and push it onto the recent list."""
    symbol = symbol.strip().upper()
    if not symbol:
        return
    st.session_state[_SYMBOL_KEY] = symbol
    recent = [item for item in st.session_state.get(_RECENT_KEY, []) if item != symbol]
    st.session_state[_RECENT_KEY] = [symbol, *recent][:MAX_RECENT]


def security_bar(services) -> str:
    """Render the pinned security header and return the loaded symbol.

    Shows the ticker box, the company's identity, its last price and where
    that price sits in the 52-week band — the four things a terminal puts
    across the top of every security screen, so the reader always knows what
    they are looking at without scrolling.
    """
    header = st.columns([1.4, 3.0, 2.0, 2.6], vertical_alignment="center")

    typed = header[0].text_input(
        "Security",
        value=loaded_symbol(),
        key=_INPUT_KEY,
        label_visibility="collapsed",
        placeholder="Ticker",
    )
    if typed.strip().upper() != loaded_symbol():
        set_symbol(typed)

    symbol = loaded_symbol()
    try:
        frame = services.security.get_history(symbol)
        quote = services.security.get_quote(symbol, frame)
    except MarketIntelError as exc:
        header[1].markdown(f":red[Could not load **{symbol}** — {exc}]")
        st.divider()
        return symbol

    fundamentals = services.security.get_fundamentals(symbol)
    currency = fundamentals.currency if fundamentals else None
    money = money_formatter(quote.price, currency)

    name = (fundamentals.name if fundamentals else None) or symbol
    descriptor = " · ".join(
        part
        for part in (
            fundamentals.sector if fundamentals else None,
            fundamentals.industry if fundamentals else None,
        )
        if part
    )
    header[1].markdown(f"**{name}**")
    header[1].caption(descriptor or f"Close of {quote.as_of:%d %b %Y}")

    change = quote.change_pct
    header[2].metric(
        "Last",
        money(quote.price),
        delta=None if change is None else f"{change:+.2f}%",
        chart_data=frame["close"].astype(float).iloc[-SPARK_SESSIONS:],
        chart_type="line",
    )
    header[3].metric(
        "52-week range",
        f"{money(quote.week52_low)} — {money(quote.week52_high)}",
        delta=_range_position(quote),
        delta_color="off",
    )

    _recent_row(symbol)
    st.divider()
    return symbol


def _range_position(quote) -> str | None:
    """How far up the 52-week band the price sits — the terminal's own framing."""
    position = quote.range_position
    return None if position is None else f"{position:.0f}% of range"


def _recent_row(current: str) -> None:
    """Clickable history, so switching between two names is one click."""
    others = [
        item for item in st.session_state.get(_RECENT_KEY, []) if item != current
    ]
    if not others:
        return
    # Fixed 10-slot grid: the buttons keep their size as the list grows rather
    # than stretching to fill the row.
    columns = st.columns([1] * len(others) + [max(1, 10 - len(others))])
    for column, symbol in zip(columns, others):
        if column.button(symbol, key=f"ae_recent_{symbol}", width="stretch"):
            set_symbol(symbol)
            st.rerun()
