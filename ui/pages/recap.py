"""Global Recap page: cross-asset snapshot plus an agentic AI market wrap.

The snapshot is expensive to build (the whole universe, one symbol at a time),
so it is held in ``st.session_state`` and only rebuilt on an explicit refresh.
Every subsequent Streamlit rerun - switching tabs, typing in the focus box -
reuses it instead of re-pricing eighty instruments.
"""

import pandas as pd
import streamlit as st

from market_intel.analysis.recap import (
    SessionState,
    level_format_for,
    top_gainers,
    top_losers,
)
from market_intel.exceptions import MarketIntelError
from market_intel.services.recap import RecapSnapshot
from market_intel.services.report import gather_headlines
from ui.context import get_services
from ui.tables import color_returns

services = get_services()

st.title("Global Recap")
st.caption(
    "Cross-asset picture by region and asset class, then a written report built "
    "from whatever you ask for."
)

_SNAPSHOT_KEY = "recap_snapshot"
_REPORT_KEY = "recap_report"
_NEWS_KEY = "recap_news"

_STATE_ICON = {
    SessionState.OPEN: "🟢",
    SessionState.CLOSED: "🔴",
    SessionState.WEEKEND: "⚪",
}


def _build_snapshot() -> RecapSnapshot:
    """Price the universe with a progress bar, and store it in session state."""
    bar = st.progress(0.0, text="Pricing the universe...")

    def on_progress(done: int, total: int, symbol: str) -> None:
        bar.progress(done / total, text=f"Pricing {symbol} ({done}/{total})")

    try:
        snapshot = services.recap.build_snapshot(progress=on_progress)
    finally:
        bar.empty()
    st.session_state[_SNAPSHOT_KEY] = snapshot
    return snapshot


def _moves_frame(moves: list) -> pd.DataFrame:
    """Build the display table for one region/asset-class block."""
    rows = []
    for move in moves:
        rows.append(
            {
                "Instrument": move.instrument.label,
                "Symbol": move.instrument.symbol,
                "Last": move.last,
                "Day %": move.pct_change,
                "Day bp": move.bp_change,
                "5D %": move.ret_5d,
                "1M %": move.ret_1m,
                "YTD %": move.ret_ytd,
                "As of": move.as_of,
                "Note": (
                    "PARTIAL SESSION - thin volume. "
                    if move.partial_session
                    else ""
                )
                + (move.instrument.note or ""),
            }
        )
    frame = pd.DataFrame(rows).set_index("Instrument")
    # Basis points only mean something for true yield quotes; drop the column
    # entirely when this block has none rather than showing a column of dashes.
    if frame["Day bp"].isna().all():
        frame = frame.drop(columns=["Day bp"])
    if (frame["Note"] == "").all():
        frame = frame.drop(columns=["Note"])
    return frame


_NUMERIC_COLUMNS = ["Day %", "Day bp", "5D %", "1M %", "YTD %"]


def _render_block(moves: list) -> None:
    """Render one asset-class table."""
    frame = _moves_frame(moves)
    config = {
        "Last": st.column_config.NumberColumn(format=level_format_for(moves)),
        "Day %": st.column_config.NumberColumn(format="%.2f"),
        "Day bp": st.column_config.NumberColumn(format="%.1f"),
        "5D %": st.column_config.NumberColumn(format="%.2f"),
        "1M %": st.column_config.NumberColumn(format="%.2f"),
        "YTD %": st.column_config.NumberColumn(format="%.2f"),
        "As of": st.column_config.DateColumn(format="YYYY-MM-DD"),
    }
    st.dataframe(
        color_returns(frame, _NUMERIC_COLUMNS),
        width="stretch",
        column_config={k: v for k, v in config.items() if k in frame.columns},
    )


# --- Controls -----------------------------------------------------------------

control_col, refresh_col = st.columns([4, 1], vertical_alignment="bottom")
with control_col:
    st.caption(
        "Snapshot is cached per session. Refresh after a session close, or when "
        "you want the latest intraday prints."
    )
with refresh_col:
    if st.button("Refresh data", width="stretch"):
        st.session_state.pop(_SNAPSHOT_KEY, None)

snapshot: RecapSnapshot | None = st.session_state.get(_SNAPSHOT_KEY)
if snapshot is None:
    try:
        snapshot = _build_snapshot()
    except MarketIntelError as exc:
        st.error(f"Could not build the snapshot: {exc}")
        st.stop()

if snapshot.is_empty:
    st.warning(
        "Nothing could be priced. The market-data provider may be unreachable - "
        "try Refresh, or check the logs."
    )
    if snapshot.failures:
        with st.expander(f"{len(snapshot.failures)} instruments failed"):
            for symbol, reason in snapshot.failures.items():
                st.text(f"{symbol}: {reason}")
    st.stop()

# --- Session clock ------------------------------------------------------------

st.subheader("Session clock")
st.caption(
    f"Snapshot built {snapshot.generated_at:%Y-%m-%d %H:%M} UTC. A region marked "
    "closed is showing its last completed session, not today's trading."
)
clock_cols = st.columns(len(snapshot.blocks))
for column, block in zip(clock_cols, snapshot.blocks):
    column.metric(
        f"{_STATE_ICON[block.state]} {block.region.value}",
        f"{block.stats.average_pct:+.2f}%",
        delta=f"{block.stats.advancers} up / {block.stats.decliners} down",
        delta_color="off",
        help=block.state_label,
    )

st.divider()

# --- Biggest movements --------------------------------------------------------

st.subheader("Biggest movements")
movers = snapshot.biggest_movers(6)
if movers:
    mover_cols = st.columns(len(movers))
    for column, move in zip(mover_cols, movers):
        column.metric(
            move.instrument.label,
            move.display_level,
            delta=move.display_change,
            help=f"{move.instrument.region.value} · {move.instrument.asset_class.value}",
        )

# Rates get their own row: basis points and percent are different units, and
# ranking a spread by percentage change puts a 4bp move above a 7% VIX spike.
rate_moves = snapshot.biggest_rate_moves(6)
if rate_moves:
    st.caption("LARGEST RATE & SPREAD MOVES (BASIS POINTS)")
    rate_cols = st.columns(len(rate_moves))
    for column, move in zip(rate_cols, rate_moves):
        column.metric(
            move.instrument.label,
            move.display_level,
            delta=move.display_change,
            delta_color="inverse",
            help=f"{move.instrument.region.value} · rising yields shown red",
        )

gain_col, lose_col = st.columns(2)
with gain_col:
    st.caption("TOP GAINERS")
    for move in top_gainers(snapshot.moves, 5):
        st.markdown(
            f"**{move.display_change}**  {move.instrument.label}  "
            f"<span style='color:#79828F'>· {move.instrument.region.value}</span>",
            unsafe_allow_html=True,
        )
with lose_col:
    st.caption("TOP LOSERS")
    for move in top_losers(snapshot.moves, 5):
        st.markdown(
            f"**{move.display_change}**  {move.instrument.label}  "
            f"<span style='color:#79828F'>· {move.instrument.region.value}</span>",
            unsafe_allow_html=True,
        )

st.divider()

# --- Top news -----------------------------------------------------------------

st.subheader("Top news")
st.caption(
    "Headlines for the day's biggest movers plus the standing benchmarks. A "
    "mover without a story is usually the one worth chasing."
)

if st.session_state.get(_NEWS_KEY) is None:
    if st.button("Load headlines"):
        with st.spinner("Collecting headlines for the movers..."):
            st.session_state[_NEWS_KEY] = gather_headlines(services.news, snapshot)
        st.rerun()
    else:
        st.caption("Not loaded yet — headlines are fetched on demand.")

articles = st.session_state.get(_NEWS_KEY)
if articles:
    news_cols = st.columns(2)
    for index, article in enumerate(articles[:12]):
        published = article.get("published_at")
        when = published.strftime("%d %b %H:%M") if published else "undated"
        source = article.get("source") or "unknown"
        symbols = ", ".join(article.get("symbols") or [])
        with news_cols[index % 2]:
            st.markdown(
                f"**[{article['headline']}]({article['url']})**  \n"
                f"<span style='color:#79828F;font-size:0.8rem'>{source} · {when}"
                + (f" · {symbols}" if symbols else "")
                + "</span>",
                unsafe_allow_html=True,
            )
elif articles == []:
    st.caption("No headlines returned for these symbols.")

# --- Region tabs --------------------------------------------------------------

st.subheader("By region and asset class")
labels = [
    f"{_STATE_ICON[block.state]} {block.region.value}" for block in snapshot.blocks
]
for tab, block in zip(st.tabs(labels), snapshot.blocks):
    with tab:
        st.caption(block.state_label)
        for asset_class, moves in block.by_asset_class.items():
            st.markdown(f"#### {asset_class.value}")
            _render_block(moves)

if snapshot.failures:
    with st.expander(f"{len(snapshot.failures)} instruments could not be priced"):
        for symbol, reason in snapshot.failures.items():
            st.text(f"{symbol}: {reason}")

st.divider()

# --- Report pipeline ----------------------------------------------------------

st.subheader("Build a report")

ai_on = services.reports.ai_available
if ai_on:
    st.caption(
        "Describe what you want and the pipeline scopes it, prices it, analyses "
        "it, and researches the news online to explain it."
    )
else:
    st.caption(
        "Describe what you want and the pipeline scopes it, prices it and writes "
        "it up. Set `MIP_ANTHROPIC_API_KEY` to add researched causation and the "
        "day's headlines - reports work without it, computed straight from the data."
    )

EXAMPLES = [
    "Full cross-asset wrap across every region",
    "Brief note on APAC equities and the yen",
    "Deep dive on rates and credit",
    "What moved in UK and Europe, and why",
]
example = st.selectbox(
    "Start from an example, or write your own below",
    options=["—"] + EXAMPLES,
    key="report_example",
)

with st.form("build_report"):
    prompt = st.text_area(
        "What report do you want?",
        value="" if example == "—" else example,
        placeholder=(
            "e.g. 'deep dive on US rates and the dollar' or "
            "'brief wrap on Asia overnight'"
        ),
        height=90,
    )
    option_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
    with option_col:
        web_search = st.checkbox(
            "Research the news online",
            value=True,
            disabled=not ai_on,
            help=(
                "Lets the model search for the headlines and data releases behind "
                "these moves. Requires an Anthropic API key."
            ),
        )
    with button_col:
        submitted = st.form_submit_button(
            "Build report", type="primary", width="stretch"
        )

if submitted:
    try:
        with st.spinner("Scoping, analysing and writing..."):
            report = services.reports.run(
                prompt, use_web_search=web_search, snapshot=snapshot
            )
        if report.articles:
            st.session_state[_NEWS_KEY] = report.articles
        st.session_state[_REPORT_KEY] = report
    except MarketIntelError as exc:
        st.error(f"Report failed: {exc}")

report = st.session_state.get(_REPORT_KEY)
if report:
    origin = (
        f"researched by Claude · {report.searches} web searches"
        if report.generated_by == "claude"
        else "computed from the snapshot, no model used"
    )
    st.caption(
        f"**{report.title}** · scope: {report.request.scope_description} · "
        f"{report.instruments_priced} instruments · {origin}"
    )
    with st.container(key="ae-prose-report"):
        st.markdown(report.content)

    st.download_button(
        "Download as Markdown",
        data=report.content,
        file_name=f"{report.title.replace(' ', '_')}.md",
        mime="text/markdown",
    )


st.divider()

# --- Research archive ---------------------------------------------------------

st.subheader("Research archive")
st.caption(
    "Every report built here is archived, alongside notes from the Research "
    "page — so you can compare what you wrote last week against today's tape."
)

query = st.text_input("Search archived reports and notes", placeholder="title, content or tag...")
notes = (
    services.research.search(query, limit=25)
    if query.strip()
    else services.research.get_recent(limit=25)
)
if not notes:
    st.caption("Nothing archived yet — build a report above and it will appear here.")
for archived in notes:
    label = f"{archived['created_at']:%Y-%m-%d %H:%M} · {archived['title']}"
    if archived["tags"]:
        label += f"  [{archived['tags']}]"
    with st.expander(label):
        origin = archived["model"] or "computed from the snapshot"
        st.caption(f"{archived['note_type']} · {origin}")
        with st.container(key=f"ae-prose-arch-{archived['id']}"):
            st.markdown(archived["content"])
