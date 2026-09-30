"""MKR 14-Framework page: one ticker, fourteen frameworks, one trade plan.

The analysis is expensive — three years of bars, an option chain across several
expiries, fundamentals, two Monte Carlo runs — so it is held in
``st.session_state`` per symbol and only rebuilt on an explicit run. Switching
tabs, toggling web search or reading the archive all reuse it.
"""

import pandas as pd
import streamlit as st

from market_intel.exceptions import MarketIntelError
from market_intel.services.mkr import MkrAnalysis, money_formatter, render_report
from ui.context import get_services
from ui.ticker import security_bar

services = get_services()

st.title("MKR 14-Framework")
st.caption(
    "Every framework that can be computed is computed — retracements, gaps, "
    "greeks, the Monte Carlo. The model is only asked for the judgement calls, "
    "and only with the numbers in front of it."
)
symbol = security_bar(services)

_ANALYSIS_KEY = "mkr_analysis"
_WRITEUP_KEY = "mkr_writeup"

_SIGNAL_ICON = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪", "n/a": "—"}


def _run(symbol: str) -> MkrAnalysis:
    """Run the analysis behind a status line and cache it in session state."""
    status = st.status(f"Analysing {symbol}...", expanded=True)

    def on_progress(message: str) -> None:
        status.write(message)

    try:
        analysis = services.mkr.analyse(symbol, progress=on_progress)
    finally:
        status.update(label=f"{symbol} analysed", state="complete", expanded=False)
    st.session_state[_ANALYSIS_KEY] = analysis
    st.session_state.pop(_WRITEUP_KEY, None)
    return analysis


# --- Input --------------------------------------------------------------------

# The ticker comes from the shared bar, but the run stays explicit: this is the
# one page that costs several seconds and a dozen requests, so loading a symbol
# elsewhere must not silently kick it off.
if st.button(f"Run analysis on {symbol}", type="primary"):
    try:
        _run(symbol)
    except MarketIntelError as exc:
        st.error(f"Could not analyse {symbol}: {exc}")

analysis: MkrAnalysis | None = st.session_state.get(_ANALYSIS_KEY)
if analysis is None:
    st.info(
        f"Run the analysis to populate the fourteen frameworks for {symbol}."
    )
    st.stop()

if analysis.symbol != symbol:
    st.warning(
        f"Showing the last run, **{analysis.symbol}**. Run the analysis again to "
        f"switch to {symbol}."
    )

# One formatter for the page, matching the report: same precision, same currency.
money = money_formatter(analysis.price, analysis.currency)

# --- Header -------------------------------------------------------------------

st.subheader(f"{analysis.symbol}" + (f" — {analysis.name}" if analysis.name else ""))
st.caption(
    f"Close of {analysis.as_of:%d %b %Y} · generated "
    f"{analysis.generated_at:%H:%M UTC} · "
    + (
        "death cross override ACTIVE"
        if analysis.confirmation.death_cross_override
        else analysis.exhaustion.summary
    )
)

header = st.columns(5)
header[0].metric(
    "Price",
    money(analysis.price),
    f"{analysis.change_pct:+.2f}%" if analysis.change_pct is not None else None,
)
header[1].metric("Confidence", f"{analysis.confidence}/100")
header[2].metric("UNI score", f"{analysis.uni.total}/25")
header[3].metric("Enter now", analysis.entry.decision)
header[4].metric(
    "52-week range",
    f"{money(analysis.week52_low)} → {money(analysis.week52_high)}"
    if analysis.week52_low and analysis.week52_high
    else "n/a",
)

if analysis.confirmation.death_cross_override:
    st.error(
        "**DEATH CROSS OVERRIDE** — the 50 EMA is below the 200 EMA and price is "
        "beneath the 50. Every bullish wave count is invalidated until price "
        f"reclaims {money(analysis.confirmation.ema50)}."
    )
elif analysis.exhaustion.triggered:
    st.warning(f"**EXHAUSTION OVERRIDE TRIGGERED** — {analysis.exhaustion.summary}.")

if analysis.unavailable:
    with st.expander(f"{len(analysis.unavailable)} data feeds unavailable"):
        for item in analysis.unavailable:
            st.text(item)

st.divider()

# --- Tabs ---------------------------------------------------------------------

scorecard_tab, levels_tab, options_tab, risk_tab, report_tab, archive_tab = st.tabs(
    ["Scorecard", "Levels & entry", "Options plan", "Risk & Monte Carlo",
     "Full report", "Archive"]
)

with scorecard_tab:
    rows = [
        {
            "Framework": signal.name,
            "Signal": f"{_SIGNAL_ICON[signal.signal]} {signal.signal}",
            "Strength": signal.strength,
            "Detail": signal.detail,
        }
        for signal in analysis.signals
    ]
    st.dataframe(
        pd.DataFrame(rows).set_index("Framework"),
        width="stretch",
        column_config={"Detail": st.column_config.TextColumn(width="large")},
    )
    bullish = sum(1 for s in analysis.signals if s.signal == "bullish")
    bearish = sum(1 for s in analysis.signals if s.signal == "bearish")
    unavailable = sum(1 for s in analysis.signals if s.signal == "n/a")
    st.caption(
        f"{bullish} bullish · {bearish} bearish · "
        f"{14 - bullish - bearish - unavailable} neutral · {unavailable} unavailable. "
        "Confidence comes from the thirteen independent reads — capital rotation is "
        "sized from it, so it is not fed back in — weights strong signals double, and "
        "is capped at 35 whenever the death cross override is in force."
    )

with levels_tab:
    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### Key price levels")
        st.markdown(
            f"🔴 **Stop:** {money(analysis.levels.stop)} (invalidation)  \n"
            f"🟡 **Support:** "
            + (" / ".join(money(x) for x in analysis.levels.supports) or "none below")
            + "  \n"
            f"🟢 **Current:** {money(analysis.price)}  \n"
            f"🎯 **Target 1:** {money(analysis.levels.target_1)} (W3/T1)  \n"
            f"🎯 **Target 2:** {money(analysis.levels.target_2)} (W5/T2)  \n"
            f"🚀 **Extended:** {money(analysis.levels.extended)}"
        )
        if analysis.levels.reward_risk:
            st.metric(
                "Reward / risk to T1",
                f"{analysis.levels.reward_risk:.2f}x",
                f"{analysis.levels.reward_pct:+.1f}% vs {analysis.levels.risk_pct:+.1f}%",
            )
        if analysis.levels.target_basis:
            st.caption(f"Targets sourced from: {analysis.levels.target_basis}.")
        if analysis.levels.atr_stop:
            st.caption(
                f"A 2x ATR stop sits at {money(analysis.levels.atr_stop)} — where "
                "structure is further away than that, size down rather than "
                "tightening into the noise."
            )

    with right:
        st.markdown("#### Entry timing")
        decision_style = {
            "Y": st.success, "Partial": st.warning, "N": st.error,
        }[analysis.entry.decision]
        decision_style(
            f"**Enter now: {analysis.entry.decision}** · trigger "
            f"{money(analysis.entry.trigger)}"
        )
        st.write(analysis.entry.rationale)
        if analysis.entry.alerts:
            st.markdown(
                "**Alerts to set:** "
                + " · ".join(
                    f"{money(level)} ({label})" for label, level in analysis.entry.alerts
                )
            )
        st.markdown("#### Capital rotation")
        rotation = analysis.rotation
        st.write(
            f"Enter **{rotation.entry_size_pct:.0f}%** of the intended position. "
            f"{rotation.scale_note} Keep at least {rotation.cash_floor_pct:.0f}% cash."
        )
        if rotation.reentry_zone:
            st.caption(
                f"Re-entry zone after taking profit: {money(rotation.reentry_zone[0])}"
                f"–{money(rotation.reentry_zone[1])}."
            )

    if analysis.fib.swing:
        st.markdown("#### Fibonacci grid")
        grid = pd.DataFrame(
            [
                {"Level": label, "Price": value, "Type": "retracement"}
                for label, value in analysis.fib.retracements.items()
            ]
            + [
                {"Level": label, "Price": value, "Type": "extension"}
                for label, value in analysis.fib.extensions.items()
            ]
        ).set_index("Level")
        st.dataframe(grid, width="stretch")
        if analysis.fib.golden_pocket:
            st.caption(
                f"Golden pocket {money(analysis.fib.golden_pocket[0])}–"
                f"{money(analysis.fib.golden_pocket[1])} · price sits at "
                f"{analysis.fib.position_pct:.0f}% of the swing · extensions anchored "
                f"on the {analysis.fib.extension_anchor}."
            )

    if analysis.fvg.unfilled:
        st.markdown("#### Unfilled fair value gaps")
        gaps = pd.DataFrame(
            [
                {
                    "Date": gap.date,
                    "Timeframe": gap.timeframe,
                    "Direction": gap.direction,
                    "From": gap.low,
                    "To": gap.high,
                    "Distance %": (gap.midpoint / analysis.price - 1.0) * 100.0,
                }
                for gap in analysis.fvg.unfilled[-12:]
            ]
        ).set_index("Date")
        st.dataframe(gaps, width="stretch")
    if not analysis.fvg.intraday_available:
        st.caption("4-hour gaps unavailable — intraday bars could not be loaded.")

with options_tab:
    leaps, wheel = analysis.leaps, analysis.wheel
    st.markdown("#### LEAPS")
    if leaps.available:
        columns = st.columns(4)
        columns[0].metric(
            "Contract", f"{leaps.expiry:%b %y} {leaps.strike:g}C"
        )
        columns[1].metric("Delta", f"{leaps.delta:.2f}")
        columns[2].metric("IV", f"{leaps.implied_vol * 100:.0f}%")
        columns[3].metric("Premium", money(leaps.premium))
        st.caption(
            f"Break-even {money(leaps.breakeven)}. Scenarios reprice the contract "
            "at the shorter maturity, so the option move is net of decay — leverage "
            "for a two-month holder, not value at expiry."
        )
        if leaps.scenarios:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Scenario": label,
                            "Stock move %": stock,
                            "Option move %": option,
                            "Action": action,
                        }
                        for label, stock, option, action in leaps.scenarios
                    ]
                ).set_index("Scenario"),
                width="stretch",
            )
        st.caption(
            f"PMCC eligible: **{'Y' if leaps.pmcc_eligible else 'N'}** — {leaps.pmcc_note}."
        )
    else:
        st.info(f"No LEAPS leg — {leaps.note}.")

    st.markdown("#### Wheel / CSP")
    if wheel.available:
        columns = st.columns(4)
        columns[0].metric("Strike", money(wheel.strike), f"{abs(wheel.delta):.2f} delta")
        columns[1].metric("Premium", money(wheel.premium))
        columns[2].metric("POP", f"~{wheel.probability_otm:.0f}%")
        columns[3].metric("Cost basis if assigned", money(wheel.cost_basis))
        st.caption(
            f"Expiry {wheel.expiry:%d %b %Y}."
            + (
                f" Covered call plan: sell the {money(wheel.covered_call_strike)} "
                "strike at T1."
                if wheel.covered_call_strike
                else ""
            )
        )
    else:
        st.info(f"No wheel leg — {wheel.note}.")

    st.markdown("#### Chain statistics")
    flow = analysis.options
    if flow.available:
        columns = st.columns(4)
        columns[0].metric(
            "Put/call OI", f"{flow.put_call_oi:.2f}" if flow.put_call_oi else "n/a"
        )
        columns[1].metric("Max pain", money(flow.max_pain))
        columns[2].metric(
            "ATM IV", f"{flow.atm_iv * 100:.0f}%" if flow.atm_iv else "n/a"
        )
        columns[3].metric(
            "IV / realised", f"{flow.iv_premium:.2f}x" if flow.iv_premium else "n/a"
        )
        st.caption(
            "IV rank is deliberately absent: no free source publishes a year of "
            "implied-vol history, so quoting one would be inventing it. The "
            "realised-vol comparison stands in for it"
            + (
                f" — 30-day realised vol is {flow.realised_vol * 100:.0f}%, the "
                f"{flow.realised_vol_percentile:.0f}th percentile of the past year."
                if flow.realised_vol and flow.realised_vol_percentile is not None
                else "."
            )
        )
        st.caption(flow.catalyst_expansion.capitalize() + ".")
        for item in flow.unusual:
            st.markdown(f"- Unusual activity: {item}")
    else:
        st.info(f"No chain statistics — {flow.note}.")

with risk_tab:
    st.markdown("#### Top 3 risks")
    for index, risk in enumerate(analysis.risks, start=1):
        st.markdown(f"{index}. {risk}")
    if not analysis.risks:
        st.caption("No structural risk flags fired — which is itself worth a check.")

    st.markdown("#### Monte Carlo")
    results = [r for r in (analysis.monte_carlo_thesis, analysis.monte_carlo_neutral) if r]
    if results:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Regime": result.label,
                        "Win rate %": result.win_rate,
                        "Avg win %": result.average_win,
                        "Avg loss %": result.average_loss,
                        "Median %": result.median_return,
                        "5th pct %": result.percentile_5,
                        "95th pct %": result.percentile_95,
                        "Worst 1% %": result.worst_1pct,
                    }
                    for result in results
                ]
            ).set_index("Regime"),
            width="stretch",
        )
        st.caption(
            f"{results[0].paths:,} paths over {results[0].horizon_days} days, "
            "bootstrapped from this security's own daily returns rather than a "
            "normal distribution, so its fat tails survive. Annualised vol "
            f"{results[-1].annual_vol:.0f}%. The thesis-tilted run recentres the "
            "drift on the path to T1 and leaves volatility alone."
        )
    else:
        st.info("Not enough history to simulate.")

    st.markdown("#### Exhaustion override")
    st.write(analysis.exhaustion.summary.capitalize() + ".")
    st.caption("Two or more flags are required before a wave top may be called.")

with report_tab:
    ai_on = services.mkr.ai_available
    if ai_on:
        st.caption(
            "The computed report is below. Have Claude research the catalysts and "
            "write the judgement layer on top of these exact numbers."
        )
    else:
        st.caption(
            "This report is computed straight from the data — no model involved. "
            "Set `MIP_ANTHROPIC_API_KEY` to add researched catalysts, the wave count "
            "in context, and the two UNI legs that need judgement."
        )

    controls = st.columns([2, 2, 1], vertical_alignment="bottom")
    with controls[0]:
        extra = st.text_input(
            "Anything the write-up should weigh (optional)",
            placeholder="e.g. I already hold a Jan 200C, sizing into earnings",
        )
    with controls[1]:
        web_search = st.checkbox(
            "Research catalysts online",
            value=True,
            disabled=not ai_on,
            help="Lets the model check the earnings calendar and recent news.",
        )
    with controls[2]:
        write = st.button(
            "Write it up", type="primary", disabled=not ai_on, width="stretch"
        )

    if write:
        try:
            with st.spinner("Researching and writing..."):
                st.session_state[_WRITEUP_KEY] = services.mkr.generate(
                    analysis, use_web_search=web_search, extra_context=extra or None
                )
        except MarketIntelError as exc:
            st.error(f"Write-up failed: {exc}")

    writeup = st.session_state.get(_WRITEUP_KEY)
    computed = render_report(analysis)
    if writeup:
        st.success(
            f"Archived as *{writeup['title']}* · {writeup['searches']} web searches "
            f"· {writeup['model']}"
        )
        with st.container(key="ae-prose-mkr"):
            st.markdown(writeup["content"])
        with st.expander("Computed report (what the model was given)"):
            with st.container(key="ae-prose-mkr-raw"):
                st.markdown(computed)
        download = writeup["content"]
    else:
        with st.container(key="ae-prose-mkr"):
            st.markdown(computed)
        download = computed

    save_col, download_col = st.columns(2)
    with save_col:
        if st.button("Archive the computed report", width="stretch"):
            note = services.mkr.archive_deterministic(analysis)
            st.success(f"Archived as *{note['title']}* (#{note['id']}).")
    with download_col:
        st.download_button(
            "Download as Markdown",
            data=download,
            file_name=f"MKR_{analysis.symbol}_{analysis.as_of:%Y%m%d}.md",
            mime="text/markdown",
            width="stretch",
        )

with archive_tab:
    st.caption(
        "Every MKR write-up is archived alongside the rest of your research, so "
        "you can read today's count against the one you made last month."
    )
    query = st.text_input(
        "Search archived analyses", value="mkr", placeholder="ticker, tag or content..."
    )
    notes = (
        services.research.search(query, limit=25)
        if query.strip()
        else services.research.get_recent(limit=25)
    )
    if not notes:
        st.caption("Nothing archived yet.")
    for archived in notes:
        label = f"{archived['created_at']:%Y-%m-%d %H:%M} · {archived['title']}"
        if archived["tags"]:
            label += f"  [{archived['tags']}]"
        with st.expander(label):
            st.caption(f"{archived['note_type']} · {archived['model'] or 'computed'}")
            with st.container(key=f"ae-prose-mkr-arch-{archived['id']}"):
                st.markdown(archived["content"])
