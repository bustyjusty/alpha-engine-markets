"""Journal page: ideas, positions and post-trade reviews."""

import pandas as pd
import streamlit as st

from market_intel.exceptions import MarketIntelError
from ui.context import get_services

services = get_services()

st.title("Trading Journal")

with st.expander("Log a new idea"):
    with st.form("log_idea", clear_on_submit=True):
        thesis = st.text_area("Thesis", placeholder="Why is this mispriced? What is the catalyst?")
        cols = st.columns(4)
        symbol = cols[0].text_input("Symbol", placeholder="optional")
        direction = cols[1].selectbox("Direction", ["long", "short"])
        target = cols[2].number_input("Target", min_value=0.0, value=0.0, step=1.0)
        stop = cols[3].number_input("Stop", min_value=0.0, value=0.0, step=1.0)
        if st.form_submit_button("Log idea", type="primary") and thesis.strip():
            try:
                services.journal.log_idea(
                    thesis.strip(),
                    symbol=symbol.strip() or None,
                    direction=direction,
                    target_price=target or None,
                    stop_price=stop or None,
                )
                st.rerun()
            except MarketIntelError as exc:
                st.error(str(exc))

status_filter = st.radio("Show", ["all", "idea", "open", "closed"], horizontal=True)
entries = services.journal.list_entries(None if status_filter == "all" else status_filter)

if not entries:
    st.info("No journal entries yet.")
    st.stop()

table = pd.DataFrame(
    [
        {
            "ID": e["id"],
            "Symbol": e["symbol"],
            "Dir": e["direction"],
            "Status": e["status"],
            "Entry": e["entry_price"],
            "Exit": e["exit_price"],
            "P&L %": e["pnl_pct"],
            "Thesis": e["thesis"],
        }
        for e in entries
    ]
).set_index("ID")
st.dataframe(
    table,
    width="stretch",
    column_config={"P&L %": st.column_config.NumberColumn(format="percent")},
)

st.subheader("Update an entry")
entry_ids = [e["id"] for e in entries]
picked = st.selectbox("Entry", entry_ids, format_func=lambda i: f"#{i}")
entry = next(e for e in entries if e["id"] == picked)

action_cols = st.columns([2, 2, 3])
if entry["status"] == "idea":
    price = action_cols[0].number_input("Entry price", min_value=0.0, step=1.0)
    if action_cols[1].button("Open position", disabled=price <= 0):
        services.journal.open_position(picked, entry_price=price)
        st.rerun()
elif entry["status"] == "open":
    price = action_cols[0].number_input("Exit price", min_value=0.0, step=1.0)
    review = action_cols[2].text_input("Review", placeholder="What worked, what didn't?")
    if action_cols[1].button("Close position", disabled=price <= 0):
        services.journal.close_position(picked, exit_price=price, review=review or None)
        st.rerun()
else:
    st.caption(f"Closed. Review: {entry['review'] or '—'}")
