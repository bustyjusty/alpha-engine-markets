"""Market Intel — Streamlit entrypoint.

Run with: streamlit run app.py
"""

import streamlit as st

from ui.style import inject_css

st.set_page_config(page_title="Market Intel", layout="wide")
inject_css()

pages = [
    st.Page("ui/pages/overview.py", title="Overview", default=True),
    st.Page("ui/pages/recap.py", title="Global Recap"),
    st.Page("ui/pages/mkr.py", title="MKR Framework"),
    st.Page("ui/pages/charts.py", title="Charts"),
    st.Page("ui/pages/news.py", title="News"),
    st.Page("ui/pages/calendar_page.py", title="Calendar"),
    st.Page("ui/pages/scanner.py", title="Scanner"),
    st.Page("ui/pages/etf_explorer.py", title="ETF Explorer"),
    st.Page("ui/pages/themes.py", title="Themes"),
    st.Page("ui/pages/research.py", title="Research"),
    st.Page("ui/pages/journal.py", title="Journal"),
]

st.navigation(pages).run()
