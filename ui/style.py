"""Global dashboard styling — Alpha Engine trading-terminal look.

Injected once from the entrypoint. Colours, fonts, radii and the semantic
palette all live in `.streamlit/config.toml`; this file only carries the
things the theme system cannot express — uppercase micro-labels, monospace
tabular figures, the panel treatment on stat tiles, and nav/tab chrome.

Shared verbatim with semi-intel and research-intelligence. Change it in one
place and copy, so the four apps keep reading as one product.

Colour discipline: chartreuse (#C2F04A) is the only accent, reserved for
primary actions and the active view. Green/red belong to signed values —
returns, deltas, candles — and are never used as decoration.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ── Canvas ──────────────────────────────────────────────────────────────
   Terminal layouts waste no vertical space, but panels need room to read. */
.block-container {
    padding-top: 2.4rem;
    padding-bottom: 3rem;
    max-width: 1480px;
}
header[data-testid="stHeader"] { background: transparent; }

/* ── Typography ──────────────────────────────────────────────────────────
   One large tight title per page; every section below it is a small
   uppercase field label, the way a terminal names its panels. */
h1 {
    letter-spacing: -0.025em;
    color: #F2F5F9;
}
h2, h3, h4 {
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: #949DAB;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #79828F;
    font-size: 0.8rem;
    line-height: 1.55;
}
hr { border-color: #1D232C; }

/* ── Long-form prose ─────────────────────────────────────────────────────
   Rendered .md files and generated research notes carry their own heading
   hierarchy. The uppercase field-label treatment above is meant for page
   sections and would mangle a forty-heading document, so anything wrapped in
   st.container(key="ae-prose...") gets ordinary document typography back. */
[class*="st-key-ae-prose"] { max-width: 62rem; }
[class*="st-key-ae-prose"] h1,
[class*="st-key-ae-prose"] h2,
[class*="st-key-ae-prose"] h3,
[class*="st-key-ae-prose"] h4 {
    text-transform: none;
    letter-spacing: -0.012em;
    color: #E7EAEF;
}
[class*="st-key-ae-prose"] h1 { font-size: 24px; }
[class*="st-key-ae-prose"] h2 { font-size: 19px; }
[class*="st-key-ae-prose"] h3 { font-size: 16px; }
[class*="st-key-ae-prose"] p,
[class*="st-key-ae-prose"] li {
    line-height: 1.72;
    color: #C4CBD5;
}

/* ── Stat tiles ──────────────────────────────────────────────────────────
   Raised panel, hairline border, and a short accent rule along the top
   edge that fades out — the one flourish borrowed from the reference. */
[data-testid="stMetric"] {
    position: relative;
    overflow: hidden;
    background: #12161C;
    border: 1px solid #232A34;
    border-radius: 12px;
    padding: 14px 16px 12px;
}
[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 1px;
    background: linear-gradient(90deg,
                rgba(194, 240, 74, 0.6), rgba(194, 240, 74, 0) 62%);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7F8896;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #F2F5F9;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 0.78rem;
}

/* Figures align wherever they appear, not just in tiles. */
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stMetric"] { font-variant-numeric: tabular-nums; }

/* ── Panels ──────────────────────────────────────────────────────────────
   Charts get the same surface as everything else so the page reads as a
   grid of instruments rather than floating widgets. */
[data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: #10141A;
    border: 1px solid #1D232C;
    border-radius: 12px;
    padding: 8px;
}

/* ── Tabs: uppercase, accent underline on the active view ────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 1.6rem;
    border-bottom: 1px solid #232A34;
}
[data-testid="stTabs"] button[role="tab"] {
    padding: 0 0 0.55rem 0;
    font-size: 0.76rem;
    font-weight: 500;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: #79828F;
}
[data-testid="stTabs"] button[aria-selected="true"] { color: #F2F5F9; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #C2F04A;
    height: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* ── Sidebar navigation ──────────────────────────────────────────────── */
[data-testid="stSidebarNav"] a {
    border-radius: 8px;
    padding-top: 0.28rem;
    padding-bottom: 0.28rem;
}
[data-testid="stSidebarNav"] a span {
    font-size: 0.76rem;
    font-weight: 500;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(194, 240, 74, 0.10);
}
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: #C2F04A; }

/* ── Controls ────────────────────────────────────────────────────────────
   The accent is bright, so primary buttons need dark text to stay legible
   (Streamlit's default is white). */
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    color: #0A0C10 !important;
    font-weight: 600;
}

/* ── Scrollbars: thin, neutral, out of the way ───────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #262E39;
    border: 2px solid transparent;
    border-radius: 999px;
    background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
    background: #3A4553;
    background-clip: content-box;
}
</style>
"""


def inject_css() -> None:
    """Apply the global stylesheet (call once per rerun from the entrypoint)."""
    st.markdown(_CSS, unsafe_allow_html=True)
