"""ETF Explorer: theme browser, holdings, ownership lookup and overlap."""

import datetime as dt

import pandas as pd
import streamlit as st

from market_intel.etf_catalog import ETF_CATALOG, all_symbols, lookup, themes
from market_intel.exceptions import MarketIntelError
from ui.context import get_services
from ui.figures import signed_bar_figure
from ui.tables import color_returns

services = get_services()

st.title("ETF Explorer")

theme_tab, holdings_tab, ownership_tab, overlap_tab = st.tabs(
    ["Themes", "Holdings", "Who owns this stock?", "Overlap"]
)

# --- Theme browser -------------------------------------------------------------
with theme_tab:
    st.caption(
        "Curated ETF universe grouped by theme. Compare performance across a "
        "theme, then pull constituent holdings into the database."
    )
    theme = st.selectbox("Theme", themes())
    entries = ETF_CATALOG[theme]
    symbols = [e.symbol for e in entries]
    names = {e.symbol: e.name for e in entries}

    action_cols = st.columns([2, 2, 4])
    compare_clicked = action_cols[0].button("Compare performance", type="primary")
    fetch_clicked = action_cols[1].button("Fetch all holdings")

    if fetch_clicked:
        with st.spinner(f"Fetching holdings for {len(symbols)} ETFs..."):
            outcomes = services.etf.refresh_many(symbols)
        fetched = [s for s, o in outcomes.items() if o == "fetched"]
        cached = [s for s, o in outcomes.items() if o == "cached"]
        failed = {s: o for s, o in outcomes.items() if o not in ("fetched", "cached")}
        if fetched:
            st.success(f"Fetched holdings: {', '.join(fetched)}")
        if cached:
            st.info(f"Already fresh (cached): {', '.join(cached)}")
        for symbol, message in failed.items():
            st.warning(f"{symbol}: {message}")

    if compare_clicked:
        start = dt.date.today() - dt.timedelta(days=200)
        rows, failures = [], []
        progress = st.progress(0.0, text="Loading prices...")
        for i, symbol in enumerate(symbols):
            progress.progress((i + 1) / len(symbols), text=f"Loading {symbol}...")
            try:
                frame = services.market_data.get_price_history(symbol, start=start)
            except MarketIntelError as exc:
                failures.append(f"{symbol}: {exc}")
                continue
            if frame.empty:
                continue
            closes = frame["adj_close"].fillna(frame["close"]).astype(float)

            def window_return(days: int) -> float | None:
                if len(closes) <= days:
                    return None
                return float(closes.iloc[-1] / closes.iloc[-1 - days] - 1.0)

            rows.append(
                {
                    "Symbol": symbol,
                    "Name": names[symbol],
                    "Last": float(closes.iloc[-1]),
                    "1D": window_return(1),
                    "1W": window_return(5),
                    "1M": window_return(21),
                    "3M": window_return(63),
                }
            )
        progress.empty()
        if failures:
            st.warning("Some ETFs failed to load:\n\n" + "\n".join(f"- {f}" for f in failures))
        if rows:
            table = pd.DataFrame(rows).set_index("Symbol")
            ranked = table.dropna(subset=["1M"]).sort_values("1M", ascending=False)
            if not ranked.empty:
                st.plotly_chart(
                    signed_bar_figure(
                        list(ranked.index),
                        [float(v) for v in ranked["1M"]],
                        title=f"{theme} — 1M total return",
                    ),
                    width="stretch",
                )
            st.dataframe(
                color_returns(table, ["1D", "1W", "1M", "3M"]),
                width="stretch",
                column_config={
                    "Last": st.column_config.NumberColumn(format="%.2f"),
                    "1D": st.column_config.NumberColumn(format="percent"),
                    "1W": st.column_config.NumberColumn(format="percent"),
                    "1M": st.column_config.NumberColumn(format="percent"),
                    "3M": st.column_config.NumberColumn(format="percent"),
                },
            )
    else:
        st.dataframe(
            pd.DataFrame([{"Symbol": e.symbol, "Name": e.name} for e in entries]),
            width="stretch",
            hide_index=True,
        )

# --- Holdings ------------------------------------------------------------------
with holdings_tab:
    cols = st.columns([2, 2, 1])
    quick = cols[0].selectbox(
        "Pick from catalog", ["(type below)"] + all_symbols(), key="holdings_quick"
    )
    typed = cols[1].text_input("ETF symbol", "SMH" if quick == "(type below)" else quick)
    etf_symbol = (typed if quick == "(type below)" else quick).strip().upper()
    if cols[2].button("Fetch / refresh", disabled=not etf_symbol):
        try:
            with st.spinner(f"Fetching holdings for {etf_symbol}..."):
                services.etf.refresh_holdings(etf_symbol)
        except MarketIntelError as exc:
            st.error(f"Could not fetch holdings: {exc}")

    if etf_symbol:
        entry = lookup(etf_symbol)
        if entry:
            st.caption(f"**{entry.symbol}** — {entry.name}")
        try:
            holdings = services.etf.get_holdings(etf_symbol)
        except MarketIntelError:
            holdings = None
        if holdings is None or holdings.empty:
            st.caption("No stored holdings yet — press *Fetch / refresh*.")
        else:
            st.caption(
                f"Top holdings as of {holdings['as_of'].iloc[0]} "
                "(Yahoo exposes the top ~10 only, so weights won't sum to 100%)."
            )
            st.plotly_chart(
                signed_bar_figure(
                    list(holdings["symbol"]),
                    [float(w or 0) for w in holdings["weight"]],
                    title=f"{etf_symbol} top holdings",
                ),
                width="stretch",
            )
            st.dataframe(
                holdings,
                width="stretch",
                hide_index=True,
                column_config={
                    "weight": st.column_config.NumberColumn("Weight", format="percent")
                },
            )

# --- Who owns this stock? -------------------------------------------------------
with ownership_tab:
    st.caption(
        "Reverse lookup: every **tracked** ETF whose latest snapshot holds the "
        "stock. Coverage grows as you fetch more ETF holdings — use "
        "*Scan catalog* to pull the whole curated universe in one go."
    )
    lookup_cols = st.columns([2, 2, 4])
    stock = lookup_cols[0].text_input("Stock symbol", "NVDA").strip().upper()
    scan_all = lookup_cols[1].button("Scan catalog", help="Fetch holdings for every catalogued ETF (one-time; cached afterwards)")

    if scan_all:
        catalog_symbols = all_symbols()
        with st.spinner(f"Fetching holdings for {len(catalog_symbols)} catalogued ETFs..."):
            outcomes = services.etf.refresh_many(catalog_symbols)
        ok = sum(1 for o in outcomes.values() if o in ("fetched", "cached"))
        st.success(f"Coverage updated: {ok}/{len(catalog_symbols)} ETFs have stored holdings data.")
        failed = {s: o for s, o in outcomes.items() if o not in ("fetched", "cached")}
        if failed:
            with st.expander(f"{len(failed)} ETFs failed (often no holdings exposed by Yahoo)"):
                for symbol, message in failed.items():
                    st.markdown(f"- **{symbol}**: {message}")

    if stock:
        exposure = services.etf.get_exposure(stock)
        if not exposure:
            st.info(
                f"No tracked ETF holds {stock} yet. Press *Scan catalog* above, or "
                "fetch specific ETFs in the Holdings tab — ownership is computed "
                "from stored snapshots."
            )
        else:
            weighted = [row for row in exposure if row["weight"] is not None]
            if weighted:
                ranked = sorted(weighted, key=lambda r: r["weight"], reverse=True)
                st.plotly_chart(
                    signed_bar_figure(
                        [row["etf"] for row in ranked],
                        [float(row["weight"]) for row in ranked],
                        title=f"Who owns {stock} — weight in each tracked ETF",
                    ),
                    width="stretch",
                )
            def etf_name(row: dict) -> str | None:
                entry = lookup(row["etf"])
                return row["etf_name"] or (entry.name if entry else None)

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ETF": row["etf"],
                            "Name": etf_name(row),
                            "Weight": row["weight"],
                            "As of": row["as_of"],
                        }
                        for row in exposure
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Weight": st.column_config.NumberColumn(format="percent"),
                },
            )

# --- Overlap -------------------------------------------------------------------
with overlap_tab:
    st.caption(
        "Common holdings between two ETFs' stored snapshots — how much of one "
        "fund you already own through the other. Fetch both in the Holdings "
        "tab first."
    )
    pair = st.columns([2, 2, 4])
    etf_a = pair[0].text_input("ETF A", "SMH").strip().upper()
    etf_b = pair[1].text_input("ETF B", "SOXX").strip().upper()
    if etf_a and etf_b and etf_a != etf_b:
        overlap = services.etf.get_overlap(etf_a, etf_b)
        if overlap.empty:
            st.info(
                f"No overlap data for {etf_a} / {etf_b}. Both need stored holdings "
                "(Holdings tab → Fetch), and Yahoo only exposes each fund's top ~10 "
                "names, so overlap here is an approximation of the true figure."
            )
        else:
            total = float(pd.to_numeric(overlap["overlap_weight"], errors="coerce").fillna(0).sum())
            tiles = st.columns(3)
            tiles[0].metric("Common holdings (top-10 basis)", f"{len(overlap)}")
            tiles[1].metric("Overlap weight", f"{total:.1%}")
            tiles[2].metric("Pair", f"{etf_a} ∩ {etf_b}")
            st.dataframe(
                overlap,
                width="stretch",
                hide_index=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "name": st.column_config.TextColumn("Name"),
                    "weight_a": st.column_config.NumberColumn(f"Wt in {etf_a}", format="percent"),
                    "weight_b": st.column_config.NumberColumn(f"Wt in {etf_b}", format="percent"),
                    "overlap_weight": st.column_config.NumberColumn("Overlap", format="percent"),
                },
            )
    elif etf_a == etf_b and etf_a:
        st.warning("Pick two different ETFs.")
