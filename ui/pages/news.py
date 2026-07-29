"""News page: collect and browse financial news."""

import streamlit as st

from ui.context import get_services

services = get_services()

st.title("News")

watchlists = services.watchlists.list_watchlists()
all_symbols = sorted({s for w in watchlists for s in w["symbols"]})

controls = st.columns([2, 2, 2])
with controls[0]:
    scope = st.selectbox("Filter by symbol", ["All"] + all_symbols)
with controls[1]:
    if st.button("Refresh news", disabled=not all_symbols):
        with st.spinner("Fetching news for watchlist symbols..."):
            stored = services.news.refresh(all_symbols)
        st.toast(f"Stored/updated {stored} articles")

if not all_symbols:
    st.info("Add symbols to a watchlist first — news is collected per watchlist symbol.")

articles = services.news.get_recent(limit=100, symbol=None if scope == "All" else scope)
if not articles:
    st.caption("No stored articles yet. Use *Refresh news* to collect some.")

for article in articles:
    published = article["published_at"]
    when = published.strftime("%Y-%m-%d %H:%M") if published else "unknown time"
    symbols = ", ".join(article["symbols"]) if article["symbols"] else ""
    st.markdown(
        f"**[{article['headline']}]({article['url']})**  \n"
        f":gray[{article['source'] or 'unknown'} · {when}"
        + (f" · {symbols}" if symbols else "")
        + "]"
    )
    if article["summary"]:
        with st.expander("Summary"):
            st.write(article["summary"])
