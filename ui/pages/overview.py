"""Overview page: watchlists with returns snapshot."""

import datetime as dt

import pandas as pd
import streamlit as st

from market_intel.exceptions import MarketIntelError
from ui.context import get_services
from ui.tables import color_returns

services = get_services()

st.title("Overview")

# --- Market pulse: benchmark tape across the top -------------------------------
PULSE = [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000"),
         ("TLT", "20Y+ Treasuries"), ("GLD", "Gold")]
pulse_tiles = st.columns(len(PULSE))
pulse_start = dt.date.today() - dt.timedelta(days=15)
for tile, (ticker, label) in zip(pulse_tiles, PULSE):
    try:
        pulse_frame = services.market_data.get_price_history(ticker, start=pulse_start)
    except MarketIntelError:
        tile.metric(label, "—")
        continue
    if pulse_frame.empty:
        tile.metric(label, "—")
        continue
    pulse_closes = pulse_frame["adj_close"].fillna(pulse_frame["close"]).astype(float)
    last = float(pulse_closes.iloc[-1])
    delta = (
        f"{last / float(pulse_closes.iloc[-2]) - 1.0:+.2%}"
        if len(pulse_closes) > 1
        else None
    )
    tile.metric(f"{label} ({ticker})", f"{last:,.2f}", delta=delta)

st.divider()

# --- Watchlist management -----------------------------------------------------
watchlists = services.watchlists.list_watchlists()
names = [w["name"] for w in watchlists] or ["Core"]

manage_col, add_col = st.columns([1, 2])
with manage_col:
    selected = st.selectbox("Watchlist", options=names, key="overview_watchlist")
with add_col:
    with st.form("add_symbol", clear_on_submit=True, border=False):
        symbol_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
        new_symbol = symbol_col.text_input("Add symbol", placeholder="e.g. NVDA")
        if button_col.form_submit_button("Add") and new_symbol.strip():
            try:
                services.watchlists.add_symbol(selected, new_symbol)
                st.rerun()
            except MarketIntelError as exc:
                st.error(f"Could not add {new_symbol.upper()}: {exc}")

symbols = services.watchlists.get_symbols(selected)
if not symbols:
    st.info("This watchlist is empty — add a symbol above to get started.")
    st.stop()

# --- Returns snapshot ---------------------------------------------------------
start = dt.date.today() - dt.timedelta(days=60)
rows, failures = [], []
for symbol in symbols:
    try:
        frame = services.market_data.get_price_history(symbol, start=start)
    except MarketIntelError as exc:
        failures.append(f"{symbol}: {exc}")
        continue
    if frame.empty:
        continue
    closes = frame["adj_close"].fillna(frame["close"]).astype(float)
    def window_return(days: int) -> float | None:
        if len(closes) <= days:
            return None
        return float(closes.iloc[-1] / closes.iloc[-1 - days] - 1.0)
    rows.append(
        {
            "Symbol": symbol,
            "Last close": float(closes.iloc[-1]),
            "1D": window_return(1),
            "5D": window_return(5),
            "1M": window_return(21),
            "As of": frame.index[-1],
        }
    )

if failures:
    st.warning("Some symbols could not be updated:\n\n" + "\n".join(f"- {f}" for f in failures))

if rows:
    # Stat tiles: the day at a glance.
    daily = [(r["Symbol"], r["1D"]) for r in rows if r["1D"] is not None]
    tiles = st.columns(4)
    tiles[0].metric("Symbols tracked", f"{len(rows)}")
    if daily:
        average = sum(move for _, move in daily) / len(daily)
        best = max(daily, key=lambda pair: pair[1])
        worst = min(daily, key=lambda pair: pair[1])
        tiles[1].metric("Avg 1D move", f"{average:+.2%}")
        tiles[2].metric("Best 1D", best[0], delta=f"{best[1]:+.2%}")
        tiles[3].metric("Worst 1D", worst[0], delta=f"{worst[1]:+.2%}")

    table = pd.DataFrame(rows).set_index("Symbol")
    st.dataframe(
        color_returns(table, ["1D", "5D", "1M"]),
        width="stretch",
        column_config={
            "Last close": st.column_config.NumberColumn(format="%.2f"),
            "1D": st.column_config.NumberColumn(format="percent"),
            "5D": st.column_config.NumberColumn(format="percent"),
            "1M": st.column_config.NumberColumn(format="percent"),
        },
    )

    remove = st.selectbox("Remove symbol", options=["—"] + symbols)
    if remove != "—" and st.button(f"Remove {remove}"):
        services.watchlists.remove_symbol(selected, remove)
        st.rerun()
