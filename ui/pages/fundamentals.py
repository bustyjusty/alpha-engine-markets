"""Fundamentals: valuation, growth, quality, ownership and the option market.

Capital IQ splits a company into tabs — Financials, Estimates, Ownership — and
that split is the useful part, because the questions are different. Valuation
asks what you are paying, growth asks what you are paying for, ownership asks
who else is in the trade, and the chain asks what the option market is pricing.

Every figure here is fetched, never derived by a model, and a field the source
does not publish reads "n/a" rather than being filled in with something
plausible. That distinction matters most exactly where the data is thinnest.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from market_intel.exceptions import MarketIntelError
from market_intel.formatting import compact_number, money_formatter, percent
from ui.context import get_services
from ui.ticker import security_bar

services = get_services()

st.title("Fundamentals")
symbol = security_bar(services)

try:
    quote = services.security.get_quote(symbol)
except MarketIntelError as exc:
    st.error(f"Could not load {symbol}: {exc}")
    st.stop()

fundamentals = services.security.get_fundamentals(symbol)
if fundamentals is None:
    st.info(
        f"No fundamentals are published for {symbol}. Indices, futures, FX and "
        "many non-US listings carry price data only — try the Charts or MKR "
        "Framework pages, which work from price history alone."
    )
    st.stop()

f = fundamentals
price = quote.price
currency = f.currency
money = money_formatter(price, currency)


def _rows(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """A two-column reference table, rendered without an index."""
    return pd.DataFrame(pairs, columns=["Metric", "Value"])


def _ratio(value: float | None, decimals: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{decimals}f}"


st.caption(
    f"{f.name or symbol} · {f.sector or 'sector n/a'} · "
    f"{f.industry or 'industry n/a'} · reported in {currency or 'an unknown currency'}"
)

valuation, growth, ownership, chain = st.tabs(
    ["Valuation", "Growth & quality", "Ownership", "Option market"]
)

# --- Valuation ----------------------------------------------------------------

with valuation:
    tiles = st.columns(4)
    tiles[0].metric("Market cap", compact_number(f.market_cap, currency))
    tiles[1].metric("Forward P/E", _ratio(f.forward_pe, 1))
    tiles[2].metric("Trailing P/E", _ratio(f.trailing_pe, 1))
    tiles[3].metric("PEG", _ratio(f.peg_ratio))

    # The PEG thresholds are the user's own screening rule, applied here so the
    # number carries the same meaning it does inside the MKR scorecard.
    if f.peg_ratio is not None:
        if f.peg_ratio < 0.5:
            st.success(
                f"PEG {f.peg_ratio:.2f} — below 0.5. Growth is being priced at a "
                "steep discount on this measure; highest-priority screen."
            )
        elif f.peg_ratio < 1.0:
            st.info(
                f"PEG {f.peg_ratio:.2f} — below 1.0, the classic growth-at-a-"
                "reasonable-price threshold."
            )

    st.dataframe(
        _rows(
            [
                ("Forward EPS", money(f.eps_forward)),
                ("Trailing EPS", money(f.eps_trailing)),
                ("Beta", _ratio(f.beta)),
                ("52-week low", money(f.fifty_two_week_low)),
                ("52-week high", money(f.fifty_two_week_high)),
                ("Analyst low target", money(f.target_low)),
                ("Analyst mean target", money(f.target_mean)),
                ("Analyst high target", money(f.target_high)),
                (
                    "Upside to mean",
                    "n/a"
                    if f.target_mean is None
                    else percent(f.target_mean / price - 1.0, signed=True),
                ),
                (
                    "Coverage",
                    "n/a" if f.analyst_count is None else f"{f.analyst_count} analysts",
                ),
                (
                    "Consensus",
                    (f.recommendation or "n/a").replace("_", " ").title(),
                ),
            ]
        ),
        hide_index=True,
        width="stretch",
    )

# --- Growth & quality ---------------------------------------------------------

with growth:
    tiles = st.columns(4)
    tiles[0].metric("Revenue growth", percent(f.revenue_growth, signed=True))
    tiles[1].metric("Earnings growth", percent(f.earnings_growth, signed=True))
    tiles[2].metric("Net margin", percent(f.profit_margin))
    tiles[3].metric(
        "Next earnings",
        "n/a" if f.next_earnings is None else f"{f.next_earnings:%d %b %Y}",
        delta=(
            None
            if f.next_earnings is None
            else f"{(f.next_earnings - dt.date.today()).days}d away"
        ),
        delta_color="off",
    )
    st.caption(
        "Growth figures are the most recent published year-over-year change, not "
        "a forecast. A blank means the source does not publish one for this "
        "security — most often a loss-maker, a recent listing, or a fund."
    )

# --- Ownership ----------------------------------------------------------------

with ownership:
    tiles = st.columns(3)
    tiles[0].metric("Held by insiders", percent(f.held_percent_insiders))
    tiles[1].metric("Held by institutions", percent(f.held_percent_institutions))
    tiles[2].metric("Short % of float", percent(f.short_percent_float))

    if f.insider_net_shares_6m is None:
        st.caption("No insider transactions filed in the last six months.")
    else:
        net = f.insider_net_shares_6m
        direction = "bought" if net > 0 else "sold"
        st.markdown(
            f"Insiders net **{direction} {abs(net):,.0f} shares** over the past six "
            "months."
        )
        st.caption(
            "Net of all filed transactions. Sales are weak evidence on their own — "
            "scheduled 10b5-1 plans and tax withholding both show up here — while "
            "open-market buying is the harder signal to explain away."
        )

# --- Option market ------------------------------------------------------------

with chain:
    options = services.security.get_options(symbol)
    if options is None:
        st.info(f"No listed options for {symbol}.")
    else:
        tiles = st.columns(4)
        tiles[0].metric("Put/call (OI)", _ratio(options.put_call_oi))
        tiles[1].metric("Put/call (volume)", _ratio(options.put_call_volume))
        tiles[2].metric("Max pain", money(options.max_pain))
        tiles[3].metric("ATM implied vol", percent(options.atm_iv))

        if options.max_pain_expiry is not None:
            st.caption(
                f"Chain statistics are taken from the "
                f"{options.max_pain_expiry:%d %b %Y} expiry — the first at least a "
                "week out, because same-day flow says nothing about positioning."
            )

        if options.unusual:
            st.markdown("**Unusual activity**")
            for line in options.unusual:
                st.markdown(f"- {line}")

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Expiry": f"{expiry:%d %b %Y}",
                        "Days out": (expiry - dt.date.today()).days,
                        "Contracts loaded": sum(
                            1 for quote in options.quotes if quote.expiry == expiry
                        ),
                    }
                    for expiry in options.expiries
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "One expiry is loaded per maturity bucket rather than the nearest few, "
            "so the weekly, monthly and LEAPS maturities are all represented."
        )
