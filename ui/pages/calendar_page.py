"""Calendar page: earnings dates and macro events."""

import datetime as dt

import pandas as pd
import streamlit as st

from ui.context import get_services

services = get_services()

st.title("Calendar")

watchlists = services.watchlists.list_watchlists()
all_symbols = sorted({s for w in watchlists for s in w["symbols"]})

top = st.columns([2, 2, 3])
days_ahead = top[0].slider("Days ahead", 7, 90, 21)
if top[1].button("Refresh earnings", disabled=not all_symbols):
    with st.spinner("Fetching earnings dates..."):
        stored = services.calendar.refresh_earnings(all_symbols)
    st.toast(f"Stored/updated {stored} events")

with st.expander("Add a macro / custom event"):
    with st.form("macro_event", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. US CPI (June)")
        cols = st.columns(2)
        event_date = cols[0].date_input("Date", value=dt.date.today())
        event_type = cols[1].selectbox("Type", ["macro", "custom"])
        notes = st.text_input("Notes", placeholder="consensus, why it matters...")
        if st.form_submit_button("Add event") and title.strip():
            services.calendar.add_manual_event(
                title.strip(),
                dt.datetime.combine(event_date, dt.time(13, 30), tzinfo=dt.timezone.utc),
                event_type=event_type,
                notes=notes or None,
            )
            st.toast("Event added")

events = services.calendar.get_upcoming(days_ahead=days_ahead, days_back=1)
if not events:
    st.info("No events in this window. Refresh earnings or add a macro event above.")
else:
    frame = pd.DataFrame(events)
    frame["when"] = pd.to_datetime(frame["when"], utc=True).dt.strftime("%a %Y-%m-%d")
    frame = frame.rename(
        columns={
            "when": "When",
            "type": "Type",
            "title": "Event",
            "symbol": "Symbol",
            "consensus": "Consensus",
            "actual": "Actual",
            "notes": "Notes",
        }
    )[["When", "Type", "Event", "Symbol", "Consensus", "Actual", "Notes"]]
    st.dataframe(frame, width="stretch", hide_index=True)
