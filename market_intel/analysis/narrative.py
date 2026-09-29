"""Turn a cross-asset snapshot into written prose, with no model involved.

A table of ninety numbers is not a market view. This module does the reading a
junior would do before the morning meeting: it finds the breadth, the leaders
and laggards, the rate moves, and - most usefully - the *divergences* where two
asset classes disagree, then states them in sentences.

Everything here is deterministic and pure. That matters for three reasons:

* The report works with **no API key**, so the dashboard is never reduced to
  tables while waiting on a credential.
* Every claim is derived arithmetically from the snapshot, so nothing in the
  deterministic sections can be fabricated.
* When the AI wrap *is* enabled, these same observations are handed to the model
  as pre-computed findings, so it spends its effort on causation and research
  rather than re-deriving what the numbers already say.
"""

from __future__ import annotations

from market_intel.analysis.recap import (
    InstrumentMove,
    SessionState,
    rank_by_magnitude,
    rank_rates_by_bp,
    top_gainers,
    top_losers,
)
from market_intel.universe import AssetClass, QuoteKind, Region

#: Percentage move above which a single instrument is worth calling out.
_NOTABLE_PCT = 1.0

#: Basis-point move above which a rate is worth calling out.
_NOTABLE_BP = 5.0

#: Share of a block that must move one way before it counts as "broad".
_BROAD_BREADTH = 0.7


#: Below this absolute percentage move an instrument is described as flat
#: rather than given a direction. Calling -0.05% "lower" overstates it.
_FLAT_PCT = 0.1


def describe_breadth(moves: list[InstrumentMove]) -> str:
    """Describe how one-sided a block's moves are, in one clause.

    Blocks of one or two instruments get no breadth language: "broadly lower
    (1 of 1 declining)" reads as a statistic but is just one number.
    """
    priced = [move for move in moves if not move.is_rate]
    if not priced:
        return "no directional read"

    if len(priced) < 3:
        only = priced[0] if len(priced) == 1 else max(priced, key=lambda m: m.magnitude)
        if abs(only.pct_change) < _FLAT_PCT:
            return "little changed"
        return "higher" if only.pct_change > 0 else "lower"

    ups = sum(1 for move in priced if move.pct_change > 0)
    total = len(priced)
    share = ups / total
    if share >= _BROAD_BREADTH:
        return f"broadly higher ({ups} of {total} advancing)"
    if share <= 1 - _BROAD_BREADTH:
        return f"broadly lower ({total - ups} of {total} declining)"
    return f"mixed ({ups} up, {total - ups} down)"


def describe_region(region: Region, state: SessionState, moves: list[InstrumentMove]) -> list[str]:
    """Write the paragraph lines for one region, asset class by asset class."""
    lines: list[str] = []
    tense = "are trading" if state is SessionState.OPEN else "closed"

    # The VIX sits in the equities block because it belongs on an equity screen,
    # but it is an implied-volatility quote, not an index. Letting it win the
    # "led at" slot would report a vol spike as equity strength.
    all_equities = _of_class(moves, AssetClass.EQUITIES)
    equities = [m for m in all_equities if m.instrument.symbol != "^VIX"]
    vix = next((m for m in all_equities if m.instrument.symbol == "^VIX"), None)

    if equities:
        leader = max(equities, key=lambda m: m.pct_change)
        laggard = min(equities, key=lambda m: m.pct_change)
        line = (
            f"**Equities** {tense} {describe_breadth(equities)}. "
            f"{leader.instrument.label} led at {leader.display_change}"
        )
        if laggard is not leader:
            line += f", {laggard.instrument.label} lagged at {laggard.display_change}"
        if vix is not None and abs(vix.pct_change) >= 3.0:
            direction = "spiked" if vix.pct_change > 0 else "fell"
            line += (
                f". Volatility {direction}: VIX {vix.display_change} to "
                f"{vix.display_level}"
            )
        lines.append(line + ".")

    rates = [m for m in _of_class(moves, AssetClass.FIXED_INCOME) if m.is_rate]
    proxies = [m for m in _of_class(moves, AssetClass.FIXED_INCOME) if not m.is_rate]
    if rates:
        widest = max(rates, key=lambda m: m.rate_magnitude)
        direction = "higher" if widest.bp_change and widest.bp_change > 0 else "lower"
        lines.append(
            f"**Fixed income** saw yields {direction}, the largest move being "
            f"{widest.instrument.label} at {widest.display_change} "
            f"(now {widest.display_level})."
        )
    elif proxies:
        widest = max(proxies, key=lambda m: m.magnitude)
        implied = "lower" if widest.pct_change > 0 else "higher"
        lines.append(
            f"**Fixed income** has no live yield on this feed; the "
            f"{widest.instrument.label} proxy moved {widest.display_change}, "
            f"implying yields {implied}."
        )

    fx = _of_class(moves, AssetClass.CURRENCIES)
    if fx:
        widest = max(fx, key=lambda m: m.magnitude)
        lines.append(
            f"**Currencies** were led by {widest.instrument.label} at "
            f"{widest.display_change} ({widest.display_level})"
            + (f" - {widest.instrument.note}." if widest.instrument.note else ".")
        )

    commodities = _of_class(moves, AssetClass.COMMODITIES)
    if commodities:
        widest = max(commodities, key=lambda m: m.magnitude)
        # "Standout" has to agree with the direction it moved, or a block
        # described as broadly higher gets illustrated with its biggest faller.
        role = "biggest gainer" if widest.pct_change > 0 else "biggest faller"
        lines.append(
            f"**Commodities** were {describe_breadth(commodities)}; the "
            f"{role} was {widest.instrument.label} at {widest.display_change} "
            f"({widest.display_level})."
        )
    return lines


def find_divergences(moves: list[InstrumentMove]) -> list[str]:
    """Flag cross-asset combinations that do not usually travel together.

    These are the observations a trader actually acts on: a risk-off equity tape
    alongside a bid in precious metals says something different from a broad
    de-risking, and a currency that will not rally on rising yields is a signal
    in itself.
    """
    findings: list[str] = []
    by_symbol = {move.instrument.symbol: move for move in moves}

    def move_of(symbol: str) -> InstrumentMove | None:
        return by_symbol.get(symbol)

    equities = [
        m
        for m in moves
        if m.instrument.asset_class is AssetClass.EQUITIES
        and m.instrument.symbol != "^VIX"
    ]
    metals = [
        by_symbol[s] for s in ("GC=F", "SI=F", "PL=F") if s in by_symbol
    ]
    rates = [m for m in moves if m.is_rate and m.bp_change is not None]

    # Equities down while precious metals rally: a debasement/haven bid rather
    # than a simple risk-off, which would normally take metals down too.
    if equities and metals:
        equity_avg = sum(m.pct_change for m in equities) / len(equities)
        metal_avg = sum(m.pct_change for m in metals) / len(metals)
        if equity_avg < -0.2 and metal_avg > 1.0:
            findings.append(
                f"Equities averaged {equity_avg:+.2f}% while precious metals "
                f"averaged {metal_avg:+.2f}% - a haven or debasement bid, not a "
                "uniform risk-off."
            )

    # Yields rising with a softer dollar is the classic fiscal-credibility tell:
    # normally higher yields pull the dollar up.
    dxy = move_of("DX-Y.NYB")
    if rates and dxy is not None:
        rate_avg = sum(m.bp_change for m in rates) / len(rates)
        if rate_avg > 2.0 and dxy.pct_change < -0.05:
            findings.append(
                f"Yields rose an average {rate_avg:+.1f}bp while the dollar index "
                f"fell {dxy.pct_change:+.2f}% - higher rates are not attracting "
                "the currency, which usually points at fiscal rather than "
                "policy-rate concerns."
            )

    # A small-cap/large-cap split is the cleanest read on domestic conditions.
    russell, spx = move_of("^RUT"), move_of("^GSPC")
    if russell is not None and spx is not None:
        gap = russell.pct_change - spx.pct_change
        if abs(gap) > 0.4:
            weaker = "Small caps lagged" if gap < 0 else "Small caps led"
            findings.append(
                f"{weaker} large caps by {abs(gap):.2f}pp "
                f"(Russell 2000 {russell.display_change} vs S&P 500 "
                f"{spx.display_change}) - the domestic-versus-global split."
            )

    # FTSE 250 versus FTSE 100 is the same test for the UK.
    ftmc, ftse = move_of("^FTMC"), move_of("^FTSE")
    if ftmc is not None and ftse is not None:
        gap = ftmc.pct_change - ftse.pct_change
        if abs(gap) > 0.4:
            findings.append(
                f"UK domestics diverged from large caps by {abs(gap):.2f}pp "
                f"(FTSE 250 {ftmc.display_change} vs FTSE 100 "
                f"{ftse.display_change})."
            )

    # Credit is the risk-appetite tell that equities alone will not give you.
    hy = move_of("FRED:HY_OAS")
    if hy is not None and hy.bp_change is not None and abs(hy.bp_change) >= 3:
        direction = "widened" if hy.bp_change > 0 else "tightened"
        implication = "risk appetite deteriorating" if hy.bp_change > 0 else "credit still bid"
        findings.append(
            f"High-yield spreads {direction} {abs(hy.bp_change):.0f}bp to "
            f"{hy.display_level} - {implication}."
        )

    # A currency that will not rally on rising domestic yields.
    usdjpy = move_of("USDJPY=X")
    if usdjpy is not None and usdjpy.pct_change > 0.3:
        findings.append(
            f"USD/JPY at {usdjpy.display_level} ({usdjpy.display_change}) - yen "
            "weakness persisting is worth watching for intervention risk."
        )
    return findings


def headline_facts(moves: list[InstrumentMove], limit: int = 3) -> list[str]:
    """The few lines that belong at the very top of the report."""
    facts: list[str] = []
    partials = [m for m in moves if m.partial_session and m.magnitude >= _NOTABLE_PCT]
    for move in rank_by_magnitude(moves, limit):
        if move.magnitude < _NOTABLE_PCT:
            continue
        facts.append(
            f"{move.instrument.label} {move.display_change} to "
            f"{move.display_level} ({move.instrument.region.value})"
        )
    for move in rank_rates_by_bp(moves, 2):
        if move.rate_magnitude >= _NOTABLE_BP:
            facts.append(
                f"{move.instrument.label} {move.display_change} to "
                f"{move.display_level}"
            )
    # Named but kept out of the ranking: a market minutes into its open can show
    # a large move on negligible volume, and silently dropping it would be as
    # misleading as ranking it first.
    for move in partials[:2]:
        facts.append(
            f"{move.instrument.label} shows {move.display_change} but is still "
            "mid-session on thin volume - not yet a comparable move"
        )
    return facts


def compose_findings(snapshot) -> str:
    """Render the deterministic findings as a Markdown block.

    Used both as the no-key report body and as pre-computed input to the AI
    wrap, so the model starts from arithmetic rather than re-deriving it.
    """
    moves = snapshot.moves
    lines: list[str] = ["## What the numbers say", ""]

    facts = headline_facts(moves)
    if facts:
        lines.append("**Headline moves**")
        lines += [f"- {fact}" for fact in facts]
        lines.append("")

    divergences = find_divergences(moves)
    if divergences:
        lines.append("**Cross-asset divergences**")
        lines += [f"- {finding}" for finding in divergences]
        lines.append("")

    for block in snapshot.blocks:
        lines.append(f"### {block.region.value} - {block.state_label}")
        region_lines = describe_region(block.region, block.state, block.moves)
        lines += region_lines if region_lines else ["No instruments priced."]
        lines.append("")

    gainers = top_gainers(moves, 5)
    losers = top_losers(moves, 5)
    if gainers or losers:
        lines.append("### Movers")
        if gainers:
            lines.append(
                "**Up:** "
                + ", ".join(f"{m.instrument.label} {m.display_change}" for m in gainers)
            )
        if losers:
            lines.append(
                "**Down:** "
                + ", ".join(f"{m.instrument.label} {m.display_change}" for m in losers)
            )
        lines.append("")

    rates = rank_rates_by_bp(moves, 6)
    if rates:
        lines.append("### Rates and spreads")
        lines += [
            f"- {m.instrument.label}: {m.display_change} to {m.display_level}"
            for m in rates
        ]
        lines.append("")
    return "\n".join(lines)


def _of_class(moves: list[InstrumentMove], asset_class: AssetClass) -> list[InstrumentMove]:
    """Filter moves down to one asset class."""
    return [move for move in moves if move.instrument.asset_class is asset_class]
