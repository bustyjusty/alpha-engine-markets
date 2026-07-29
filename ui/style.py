"""Global dashboard styling — Bloomberg-terminal look.

Injected once from app.py; base colors live in .streamlit/config.toml,
this adds the panel treatment: black surfaces, amber field labels and
headers, hairline borders, dense tabular numerics. Green/red is reserved
for signed values (returns, candles), never decoration.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Denser canvas: terminal layouts waste no vertical space */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

/* Stat tiles: black panels, amber field labels, white values */
[data-testid="stMetric"] {
    background: #101010;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 4px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] {
    color: #fb8b1e;
    font-size: 0.70rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-variant-numeric: tabular-nums;
    font-weight: 650;
    font-size: 1.5rem;
    color: #f2f0eb;
}
[data-testid="stMetricDelta"] {
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
}

/* Tables and charts sit on black panels with hairline borders */
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"] {
    background: #0d0d0d;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 4px;
    padding: 6px;
}

/* Expanders and forms as flat panels */
[data-testid="stExpander"],
[data-testid="stForm"] {
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 4px;
    background: #0d0d0d;
}

/* Page titles as terminal function headers: uppercase amber */
h1 {
    color: #fb8b1e;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-weight: 700;
    font-size: 1.25rem;
}
h2, h3 {
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 1.0rem;
    color: #d9d6cf;
}

/* Section captions: quiet gray */
[data-testid="stCaptionContainer"] { color: #8f8d86; }

/* Tabs: amber underline on the active tab */
[data-testid="stTabs"] button[aria-selected="true"] { color: #fb8b1e; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #fb8b1e; }
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: rgba(255, 255, 255, 0.10);
}

/* Sidebar nav: darkest surface, uppercase entries */
[data-testid="stSidebar"] {
    background: #050505;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
[data-testid="stSidebarNav"] a span {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.82rem;
}

/* Numbers in tables align */
[data-testid="stDataFrame"] * { font-variant-numeric: tabular-nums; }

/* Primary buttons: amber with black text for contrast */
button[kind="primary"] { color: #000000 !important; font-weight: 650; }
</style>
"""


def inject_css() -> None:
    """Apply the global stylesheet (call once per rerun from the entrypoint)."""
    st.markdown(_CSS, unsafe_allow_html=True)
