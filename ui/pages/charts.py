"""Charts page: candlestick + volume with technical indicator overlays."""

import datetime as dt

import streamlit as st

from market_intel.analysis import bollinger, rolling_twap, rolling_vwap, rsi, sma
from market_intel.exceptions import MarketIntelError
from ui.context import get_services
from ui.figures import price_figure

services = get_services()

st.title("Charts")

controls = st.columns([2, 2, 4])
symbol = controls[0].text_input("Symbol", value="SPY").strip().upper()
lookback = controls[1].selectbox("Lookback", ["3M", "6M", "1Y", "2Y"], index=2)
days = {"3M": 92, "6M": 183, "1Y": 365, "2Y": 730}[lookback]
with controls[2]:
    indicator_picks = st.multiselect(
        "Indicators",
        ["VWAP (20)", "TWAP (20)", "SMA 20", "SMA 50", "Bollinger (20, 2σ)", "RSI (14)"],
        default=["VWAP (20)", "SMA 50", "RSI (14)"],
    )

if symbol:
    try:
        frame = services.market_data.get_price_history(
            symbol, start=dt.date.today() - dt.timedelta(days=days)
        )
    except MarketIntelError as exc:
        st.error(f"Could not load {symbol}: {exc}")
        st.stop()

    if frame.empty:
        st.info(f"No price data available for {symbol}.")
        st.stop()

    closes = frame["close"].astype(float)

    overlays = {}
    if "VWAP (20)" in indicator_picks:
        overlays["VWAP"] = rolling_vwap(frame, 20)
    if "TWAP (20)" in indicator_picks:
        overlays["TWAP"] = rolling_twap(frame, 20)
    if "SMA 20" in indicator_picks:
        overlays["SMA 20"] = sma(closes, 20)
    if "SMA 50" in indicator_picks:
        overlays["SMA 50"] = sma(closes, 50)
    bands = None
    if "Bollinger (20, 2σ)" in indicator_picks:
        upper, _, lower = bollinger(closes)
        bands = (upper, lower)
    rsi_series = rsi(closes) if "RSI (14)" in indicator_picks else None

    st.plotly_chart(
        price_figure(frame, symbol, overlays=overlays, bands=bands, rsi_series=rsi_series),
        width="stretch",
    )

    adj = frame["adj_close"].fillna(frame["close"]).astype(float)
    total_return = adj.iloc[-1] / adj.iloc[0] - 1.0
    latest_rsi = rsi(closes).iloc[-1]
    latest_vwap = rolling_vwap(frame, 20).iloc[-1]

    tiles = st.columns(5)
    tiles[0].metric("Last close", f"{closes.iloc[-1]:,.2f}")
    tiles[1].metric(f"{lookback} return", f"{total_return:+.1%}")
    tiles[2].metric("RSI (14)", f"{latest_rsi:.0f}" if latest_rsi == latest_rsi else "n/a")
    if latest_vwap == latest_vwap:  # not NaN
        deviation = closes.iloc[-1] / latest_vwap - 1.0
        tiles[3].metric("vs VWAP (20)", f"{deviation:+.1%}")
    else:
        tiles[3].metric("vs VWAP (20)", "n/a")
    tiles[4].metric(f"{lookback} range", f"{frame['low'].min():,.0f}–{frame['high'].max():,.0f}")
