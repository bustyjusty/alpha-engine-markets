"""Scanner page: relative-value scan of a universe vs a benchmark."""

import streamlit as st

from ui.context import get_services
from ui.figures import signed_bar_figure

services = get_services()

st.title("Relative-Value Scanner")
st.caption(
    "Ranks names by how unusual their recent move is **relative to a benchmark**, "
    "z-scored against their own history. Large |z| flags candidates for review — "
    "it is not a trade signal."
)

watchlist_names = [w["name"] for w in services.watchlists.list_watchlists()]
theme_names = [t["name"] for t in services.themes.list_themes()]

controls = st.columns([2, 2, 2, 2])
source = controls[0].selectbox("Universe", ["Watchlist", "Theme", "Custom"])
if source == "Watchlist":
    picked = controls[1].selectbox("Watchlist", watchlist_names or ["—"])
    symbols = services.watchlists.get_symbols(picked) if watchlist_names else []
elif source == "Theme":
    picked = controls[1].selectbox("Theme", theme_names or ["—"])
    symbols = services.themes.get_symbols(picked) if theme_names else []
else:
    raw = controls[1].text_input("Symbols (comma-separated)", "NVDA, AMD, INTC, TSM")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

benchmark = controls[2].text_input("Benchmark", "SPY").strip().upper()
window = controls[3].slider("Reaction window (days)", 3, 20, 5)

if st.button("Run scan", type="primary", disabled=not symbols):
    with st.spinner(f"Scanning {len(symbols)} symbols vs {benchmark}..."):
        result = services.scanner.scan(symbols, benchmark_symbol=benchmark, window_days=window)

    if result.empty:
        st.error("Scan produced no results — check the benchmark symbol and universe.")
    else:
        flagged = result[
            (result["signal"].isin(["overreaction", "underreaction"]))
            | (result["signals"].astype(str) != "")
        ]
        if not flagged.empty:
            st.subheader("Flagged")
            for symbol, row in flagged.iterrows():
                parts = []
                if row["signal"] == "overreaction":
                    parts.append(
                        f"▲ over-reaction vs benchmark (z = {row['zscore']:+.2f}, "
                        f"{window}D rel. {row['relative_return']:+.1%})"
                    )
                elif row["signal"] == "underreaction":
                    parts.append(
                        f"▼ under-reaction vs benchmark (z = {row['zscore']:+.2f}, "
                        f"{window}D rel. {row['relative_return']:+.1%})"
                    )
                if row["signals"]:
                    parts.append(row["signals"])
                extras = []
                if row["rsi"] is not None:
                    extras.append(f"RSI {row['rsi']:.0f}")
                if row["vwap_dev"] is not None:
                    extras.append(f"{row['vwap_dev']:+.1%} vs VWAP")
                suffix = f" :gray[· {' · '.join(extras)}]" if extras else ""
                st.markdown(f"**{symbol}** — " + " · ".join(parts) + suffix)
        else:
            st.info("Nothing flagged — no unusual relative moves or indicator signals in this window.")

        scored = result.dropna(subset=["zscore"])
        if not scored.empty:
            st.plotly_chart(
                signed_bar_figure(
                    list(scored.index),
                    [float(z) for z in scored["zscore"]],
                    title=f"{window}D relative-move z-score vs {benchmark}",
                    value_format=".1f",
                ),
                width="stretch",
            )
        st.dataframe(
            result,
            width="stretch",
            column_config={
                "window_return": st.column_config.NumberColumn("Window return", format="percent"),
                "benchmark_return": st.column_config.NumberColumn("Benchmark", format="percent"),
                "relative_return": st.column_config.NumberColumn("Relative", format="percent"),
                "zscore": st.column_config.NumberColumn("Z-score", format="%.2f"),
                "signal": st.column_config.TextColumn("RV signal"),
                "last_close": st.column_config.NumberColumn("Last", format="%.2f"),
                "rsi": st.column_config.NumberColumn("RSI 14", format="%.0f"),
                "vwap_dev": st.column_config.NumberColumn("vs VWAP", format="percent"),
                "percent_b": st.column_config.NumberColumn("%B", format="%.2f"),
                "signals": st.column_config.TextColumn("Indicator signals", width="medium"),
            },
        )
