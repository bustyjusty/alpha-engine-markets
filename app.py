"""Market Intel — Streamlit entrypoint.

Run with: streamlit run app.py

Navigation is grouped by the question being asked, not by the tool doing the
asking. MARKETS is top-down and needs no ticker. SECURITY is everything about
one loaded name, so those pages share the ticker bar rather than each asking
for the symbol again. SCREENING is the search for candidates, and WORKSPACE is
where the output of all of it gets written down.
"""

import streamlit as st

from market_intel._build import BUILD
from ui.style import inject_css

st.set_page_config(page_title="Market Intel", layout="wide")
inject_css()

pages = {
    "Markets": [
        st.Page("ui/pages/overview.py", title="Overview", icon=":material/dashboard:",
                default=True),
        st.Page("ui/pages/recap.py", title="Global Recap", icon=":material/public:"),
        st.Page("ui/pages/calendar_page.py", title="Calendar",
                icon=":material/event:"),
        st.Page("ui/pages/news.py", title="News", icon=":material/newspaper:"),
    ],
    "Security": [
        st.Page("ui/pages/snapshot.py", title="Snapshot",
                icon=":material/contact_page:"),
        st.Page("ui/pages/fundamentals.py", title="Fundamentals",
                icon=":material/account_balance:"),
        st.Page("ui/pages/charts.py", title="Charts",
                icon=":material/candlestick_chart:"),
        st.Page("ui/pages/mkr.py", title="MKR Framework",
                icon=":material/insights:"),
    ],
    "Screening": [
        st.Page("ui/pages/scanner.py", title="Scanner", icon=":material/radar:"),
        st.Page("ui/pages/themes.py", title="Themes", icon=":material/category:"),
        st.Page("ui/pages/etf_explorer.py", title="ETF Explorer",
                icon=":material/donut_small:"),
    ],
    "Workspace": [
        st.Page("ui/pages/research.py", title="Research",
                icon=":material/science:"),
        st.Page("ui/pages/journal.py", title="Journal",
                icon=":material/edit_note:"),
    ],
}

navigation = st.navigation(pages)

# Which build is actually being served. Streamlit Cloud announces nothing when
# a rebuild lands, and a cached page is indistinguishable from a fresh one, so
# the stamp is the only way to tell a deployed change from a stale tab.
st.sidebar.caption(f"Build {BUILD}")

navigation.run()
