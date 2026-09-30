"""Snapshot: the one screen that says what a security is and what it is doing.

The terminal equivalent of Bloomberg's DES or a Capital IQ company tearsheet —
identity, performance across every horizon, the price with its two structural
averages, where the street thinks it is going, and what is coming up. Anything
needing a framework belongs on another page; this one is for orientation.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from market_intel.analysis import sma
from market_intel.exceptions import MarketIntelError
from market_intel.formatting import compact_number, money_formatter, percent
from ui.context import get_services
from ui.figures import price_figure
from ui.ticker import security_bar

#: Horizons the performance strip reports, in trading-day terms.
_HORIZONS: tuple[tuple[str, int], ...] = (
    ("1D", 1),
    ("1W", 5),
    ("1M", 21),
    ("3M", 63),
    ("6M", 126),
    ("1Y", 252),
)

services = get_services()

st.title("Snapshot")
symbol = security_bar(services)

try:
    frame = services.security.get_history(symbol, days=500)
except MarketIntelError as exc:
    st.error(f"Could not load {symbol}: {exc}")
    st.stop()

if frame.empty:
    st.info(f"No price data available for {symbol}.")
    st.stop()

fundamentals = services.security.get_fundamentals(symbol)
closes = frame["close"].astype(float)
price = float(closes.iloc[-1])
currency = fundamentals.currency if fundamentals else None
money = money_formatter(price, currency)


def _return_over(sessions: int) -> float | None:
    """Total return over the trailing N sessions, or None if history is short."""
    if len(closes) <= sessions:
        return None
    return closes.iloc[-1] / closes.iloc[-1 - sessions] - 1.0


def _ytd_return() -> float | None:
    """Return since the last close of the previous calendar year.

    The stored index holds plain dates, so it is normalised to timestamps
    before the comparison rather than relying on object-dtype ordering.
    """
    index = pd.to_datetime(closes.index)
    prior = closes[index < pd.Timestamp(index[-1].year, 1, 1)]
    return closes.iloc[-1] / prior.iloc[-1] - 1.0 if len(prior) else None


# --- Performance --------------------------------------------------------------

st.subheader("Performance")
horizons = [*_HORIZONS, ("YTD", 0)]
tiles = st.columns(len(horizons))
for tile, (label, sessions) in zip(tiles, horizons):
    value = _ytd_return() if label == "YTD" else _return_over(sessions)
    tile.metric(label, "n/a" if value is None else percent(value, signed=True))

# --- Price --------------------------------------------------------------------

st.subheader("Price")
overlays = {}
if len(closes) >= 50:
    overlays["SMA 50"] = sma(closes, 50)
if len(closes) >= 200:
    overlays["SMA 200"] = sma(closes, 200)
st.plotly_chart(price_figure(frame, symbol, overlays=overlays), width="stretch")

# --- Reference data -----------------------------------------------------------

if fundamentals is None:
    st.subheader("Key facts")
    st.caption(
        f"No fundamentals are published for {symbol}. Indices, futures and many "
        "non-US listings carry price data only — the performance and price "
        "sections above are complete."
    )
    st.stop()

f = fundamentals

st.subheader("Key facts")
facts = st.columns(5)
facts[0].metric("Market cap", compact_number(f.market_cap, currency))
facts[1].metric("Forward P/E", "n/a" if f.forward_pe is None else f"{f.forward_pe:,.1f}")
facts[2].metric("PEG", "n/a" if f.peg_ratio is None else f"{f.peg_ratio:,.2f}")
facts[3].metric("Beta", "n/a" if f.beta is None else f"{f.beta:,.2f}")
facts[4].metric(
    "Next earnings",
    "n/a" if f.next_earnings is None else f"{f.next_earnings:%d %b}",
    delta=(
        None
        if f.next_earnings is None
        else f"{(f.next_earnings - dt.date.today()).days}d"
    ),
    delta_color="off",
)

# --- The street ---------------------------------------------------------------

st.subheader("Analyst targets")
if f.target_mean is None:
    st.caption("No published price targets for this security.")
else:
    upside = f.target_mean / price - 1.0
    targets = st.columns([1, 1, 1, 1.4])
    targets[0].metric("Low", money(f.target_low))
    targets[1].metric("Mean", money(f.target_mean), delta=percent(upside, signed=True))
    targets[2].metric("High", money(f.target_high))
    targets[3].metric(
        "Coverage",
        "n/a" if f.analyst_count is None else f"{f.analyst_count} analysts",
        delta=(f.recommendation or "").replace("_", " ").title() or None,
        delta_color="off",
    )
    if f.target_low is not None and f.target_high is not None:
        st.progress(
            min(max((price - f.target_low) / (f.target_high - f.target_low), 0.0), 1.0)
            if f.target_high > f.target_low
            else 0.0,
            text=f"{money(price)} within the {money(f.target_low)} — "
            f"{money(f.target_high)} target range",
        )

# --- Recent news --------------------------------------------------------------

st.subheader("Recent news")
articles = services.news.get_recent(limit=6, symbol=symbol)
if not articles:
    st.caption(
        f"Nothing stored for {symbol}. The News page collects headlines for "
        "watchlist symbols — add this one to a watchlist and refresh."
    )
for article in articles:
    published = article["published_at"]
    when = published.strftime("%d %b %H:%M") if published else "unknown time"
    st.markdown(
        f"**[{article['headline']}]({article['url']})**  \n"
        f":gray[{article['source'] or 'unknown'} · {when}]"
    )
