"""Research page: AI-generated notes and the research archive."""

import streamlit as st

from market_intel.exceptions import MarketIntelError
from market_intel.services.research import compose_market_context
from ui.context import get_services

services = get_services()

st.title("Research")

generate_tab, archive_tab = st.tabs(["Generate", "Archive"])

with generate_tab:
    if not services.research.available:
        st.warning(
            "AI research is disabled — set `MIP_ANTHROPIC_API_KEY` in your `.env` "
            "to enable note generation. The archive below still works."
        )
    with st.form("generate_note"):
        title = st.text_input("Note title", value="Market wrap")
        sources = st.multiselect(
            "Context to include",
            ["Recent news", "Upcoming events"],
            default=["Recent news", "Upcoming events"],
        )
        focus = st.text_input("Focus (optional)", placeholder="e.g. semiconductors, rates")
        tags = st.text_input("Tags (optional, comma-separated)", placeholder="semis, macro")
        submitted = st.form_submit_button(
            "Generate note", type="primary", disabled=not services.research.available
        )

    if submitted:
        context = compose_market_context(
            news=services.news.get_recent(limit=25) if "Recent news" in sources else None,
            events=services.calendar.get_upcoming(days_ahead=14)
            if "Upcoming events" in sources
            else None,
        )
        try:
            with st.spinner("Generating research note..."):
                note = services.research.generate_note(
                    title, context=context, focus=focus or None, tags=tags or None
                )
            st.success(f"Note archived (#{note['id']}, model {note['model']})")
            # Keyed container so the stylesheet restores document-scale heading
            # typography — generated notes carry their own heading hierarchy.
            with st.container(key="ae-prose-new"):
                st.markdown(note["content"])
        except MarketIntelError as exc:
            st.error(f"Generation failed: {exc}")

    with st.expander("Add a manual note instead"):
        with st.form("manual_note", clear_on_submit=True):
            manual_title = st.text_input("Title")
            manual_content = st.text_area("Content (Markdown)", height=200)
            manual_tags = st.text_input("Tags", placeholder="optional")
            if st.form_submit_button("Save") and manual_title.strip() and manual_content.strip():
                services.research.add_manual_note(
                    manual_title.strip(), manual_content, tags=manual_tags or None
                )
                st.toast("Note saved")

with archive_tab:
    query = st.text_input("Search the archive", placeholder="title, content or tag...")
    notes = services.research.search(query) if query.strip() else services.research.get_recent()
    if not notes:
        st.caption("No notes found.")
    for note in notes:
        label = f"{note['created_at']:%Y-%m-%d} · {note['title']}"
        if note["tags"]:
            label += f"  [{note['tags']}]"
        with st.expander(label):
            st.caption(f"{note['note_type']}" + (f" · {note['model']}" if note["model"] else ""))
            with st.container(key=f"ae-prose-{note['id']}"):
                st.markdown(note["content"])
