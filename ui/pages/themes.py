"""Themes page: track baskets like AI, semis, defence, energy."""

import streamlit as st

from market_intel.exceptions import MarketIntelError
from ui.context import get_services
from ui.figures import signed_bar_figure

services = get_services()

st.title("Themes")

with st.expander("Create theme / add member"):
    create_col, member_col = st.columns(2)
    with create_col, st.form("create_theme", clear_on_submit=True, border=False):
        name = st.text_input("New theme name", placeholder="e.g. AI infrastructure")
        description = st.text_input("Description", placeholder="optional")
        if st.form_submit_button("Create theme") and name.strip():
            services.themes.create_theme(name.strip(), description or None)
            st.rerun()
    theme_names = [t["name"] for t in services.themes.list_themes()]
    with member_col, st.form("add_member", clear_on_submit=True, border=False):
        target = st.selectbox("Theme", theme_names or ["—"])
        symbol = st.text_input("Symbol", placeholder="e.g. NVDA")
        note = st.text_input("Why it belongs", placeholder="optional rationale")
        if st.form_submit_button("Add member") and theme_names and symbol.strip():
            try:
                services.themes.add_member(target, symbol, note=note or None)
                st.rerun()
            except MarketIntelError as exc:
                st.error(str(exc))

themes = services.themes.list_themes()
if not themes:
    st.info("No themes yet — create one above (e.g. AI, semiconductors, defence, energy).")
    st.stop()

days = st.slider("Performance window (days)", 7, 180, 30)

for theme in themes:
    st.subheader(theme["name"])
    if theme["description"]:
        st.caption(theme["description"])
    if not theme["members"]:
        st.caption("No members yet.")
        continue

    with st.spinner(f"Computing {theme['name']} performance..."):
        perf = services.themes.performance(theme["name"], days=days)
    if perf:
        st.plotly_chart(
            signed_bar_figure(
                [row["symbol"] for row in perf],
                [row["return"] for row in perf],
                title=f"{days}D return",
            ),
            width="stretch",
        )
    notes = [m for m in theme["members"] if m["note"]]
    if notes:
        with st.expander("Member rationale"):
            for member in notes:
                st.markdown(f"- **{member['symbol']}** — {member['note']}")
