"""Prompt-driven report pipeline: one instruction in, a written report out.

The pipeline is five explicit stages, each independently testable:

1. **Interpret** - read the free-text prompt and work out what it is asking for
   (which regions, which asset classes, how long the answer should be).
2. **Gather** - price the requested slice of the universe into a snapshot.
3. **Analyse** - derive the deterministic findings: breadth, leaders, laggards,
   rate moves and cross-asset divergences.
4. **Write** - either hand the findings to Claude with web search enabled, or,
   when no API key is configured, render the findings directly as prose.
5. **Archive** - store the finished report alongside the research notes.

Stage 4 is the important design decision. The deterministic writer means a
report is *always* produced: without a key the user gets real prose rather than
a table and an apology. With a key, the same findings are handed to the model as
pre-computed arithmetic so its effort goes into researching causation instead of
re-deriving what the snapshot already states.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from market_intel.analysis.narrative import compose_findings
from market_intel.universe import AssetClass, Region

logger = logging.getLogger(__name__)

#: Words a user might type mapped to the region they mean.
_REGION_KEYWORDS: dict[Region, tuple[str, ...]] = {
    Region.APAC: (
        "apac", "asia", "asian", "japan", "japanese", "china", "chinese", "hong kong",
        "korea", "korean", "taiwan", "australia", "india", "indian", "singapore",
        "nikkei", "hang seng", "kospi", "asx", "nifty", "yen", "yuan", "won",
    ),
    Region.UK: ("uk", "britain", "british", "london", "gilt", "gilts", "ftse", "sterling", "gbp", "boe"),
    Region.EUROPE: (
        "europe", "european", "eurozone", "euro", "germany", "german", "dax",
        "france", "french", "cac", "italy", "italian", "bund", "ecb", "stoxx",
    ),
    Region.US: (
        "us", "usa", "america", "american", "united states", "wall street", "fed",
        "treasury", "treasuries", "s&p", "sp500", "nasdaq", "dow", "russell", "dollar",
    ),
    Region.GLOBAL: ("global", "commodity", "commodities", "gold", "silver", "oil", "metals"),
}

#: Words mapped to the asset class they refer to.
_ASSET_CLASS_KEYWORDS: dict[AssetClass, tuple[str, ...]] = {
    AssetClass.EQUITIES: ("equity", "equities", "stock", "stocks", "shares", "index", "indices"),
    AssetClass.FIXED_INCOME: (
        "bond", "bonds", "rates", "yield", "yields", "fixed income", "credit",
        "spread", "spreads", "duration", "curve", "treasury", "treasuries", "gilt", "bund",
    ),
    AssetClass.CURRENCIES: ("fx", "currency", "currencies", "forex", "dollar", "yen", "euro", "sterling"),
    AssetClass.COMMODITIES: (
        "commodity", "commodities", "oil", "crude", "brent", "wti", "gold",
        "silver", "copper", "metals", "gas", "wheat",
    ),
}

_BRIEF_WORDS = ("brief", "short", "quick", "summary", "tldr", "headline", "one-liner", "concise")
_DEEP_WORDS = ("deep", "detailed", "thorough", "comprehensive", "full", "in-depth", "elaborate")


@dataclass(frozen=True, slots=True)
class ReportRequest:
    """What the user asked for, after interpreting their prompt.

    Attributes:
        prompt: The original free-text instruction, passed through to the model.
        regions: Regions to price. Empty means the whole universe.
        asset_classes: Asset classes to emphasise. Empty means all of them.
        depth: ``brief``, ``standard`` or ``deep`` - drives length and effort.
        use_web_search: Whether the model may research the news online.
    """

    prompt: str
    regions: tuple[Region, ...] = ()
    asset_classes: tuple[AssetClass, ...] = ()
    depth: str = "standard"
    use_web_search: bool = True

    @property
    def scope_description(self) -> str:
        """Human-readable summary of the resolved scope, shown in the UI."""
        regions = ", ".join(r.value for r in self.regions) if self.regions else "all regions"
        classes = (
            ", ".join(a.value for a in self.asset_classes)
            if self.asset_classes
            else "all asset classes"
        )
        return f"{regions} · {classes} · {self.depth}"


@dataclass(frozen=True, slots=True)
class Report:
    """A finished report and the trail of how it was produced.

    Attributes:
        title: Archive title.
        content: The report body, in Markdown.
        request: The interpreted request that produced it.
        generated_by: ``claude`` or ``deterministic``.
        searches: Number of web searches the model ran, if any.
        instruments_priced: How many instruments the snapshot covered.
        failures: Instruments that could not be priced.
        articles: Headlines gathered for the movers and benchmarks.
        note_id: Row id in the research archive, when archived.
    """

    title: str
    content: str
    request: ReportRequest
    generated_by: str
    searches: int = 0
    instruments_priced: int = 0
    failures: dict[str, str] = field(default_factory=dict)
    articles: list[dict] = field(default_factory=list)
    note_id: int | None = None


def interpret_prompt(prompt: str, use_web_search: bool = True) -> ReportRequest:
    """Read a free-text prompt and resolve it into a concrete request.

    Keyword matching is deliberately generous: a prompt mentioning "the yen" is
    asking about APAC and currencies even though it names neither. Anything the
    prompt does not pin down is left empty, which downstream means "everything"
    rather than a guessed narrowing - a report that silently dropped Europe
    because the user said "dollar" would be worse than one that included it.

    Args:
        prompt: The user's instruction. An empty prompt yields a full-scope
            standard report.
        use_web_search: Whether online research is permitted.
    """
    text = (prompt or "").lower()

    regions = tuple(
        region
        for region, keywords in _REGION_KEYWORDS.items()
        if any(_mentions(text, word) for word in keywords)
    )
    asset_classes = tuple(
        asset_class
        for asset_class, keywords in _ASSET_CLASS_KEYWORDS.items()
        if any(_mentions(text, word) for word in keywords)
    )

    depth = "standard"
    if any(_mentions(text, word) for word in _DEEP_WORDS):
        depth = "deep"
    elif any(_mentions(text, word) for word in _BRIEF_WORDS):
        depth = "brief"

    return ReportRequest(
        prompt=prompt or "",
        regions=regions,
        asset_classes=asset_classes,
        depth=depth,
        use_web_search=use_web_search,
    )


def _mentions(text: str, phrase: str) -> bool:
    """Whether ``phrase`` appears in ``text`` as a whole word or phrase."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


#: Length and emphasis guidance handed to the model, per depth.
_DEPTH_GUIDANCE = {
    "brief": (
        "Keep this tight: a headline paragraph, then at most one short "
        "paragraph per region. Aim for something readable in under a minute."
    ),
    "standard": (
        "Give a full desk note: a 'what matters today' opener, a section per "
        "region broken down by asset class, the biggest movers, and what to "
        "watch next."
    ),
    "deep": (
        "Go deep: cover every region and asset class, quantify the divergences, "
        "research the underlying catalysts thoroughly, discuss positioning and "
        "second-order effects, and close with a detailed watch list."
    ),
}


#: How many of the day's biggest movers to pull headlines for. Each is a
#: network round-trip, so this trades breadth against how long a report takes.
_NEWS_SYMBOL_LIMIT = 8

#: Benchmarks always worth headlines regardless of whether they moved, because
#: they carry the day's general market story rather than a single name's.
_NEWS_ANCHORS = ("^GSPC", "^N225", "^FTSE", "GC=F", "CL=F")


def gather_headlines(
    news_service,
    snapshot,
    per_symbol: int = 4,
    limit: int = 25,
) -> list[dict]:
    """Collect headlines for the day's movers plus the standing benchmarks.

    News is fetched for the instruments that actually moved, on the theory that
    a mover without a story is the thing worth chasing, and a story without a
    move usually is not. A handful of anchors are always included so the report
    still carries the general market narrative on a quiet day.

    Failures are swallowed: a report missing its headlines is a worse outcome
    than a report that never renders, so news is strictly best-effort.

    Returns:
        Article dicts, newest first, de-duplicated by URL.
    """
    if news_service is None:
        return []

    movers = [move.instrument.symbol for move in snapshot.biggest_movers(_NEWS_SYMBOL_LIMIT)]
    priced = {move.instrument.symbol for move in snapshot.moves}
    symbols: list[str] = []
    for symbol in list(movers) + [s for s in _NEWS_ANCHORS if s in priced]:
        if symbol not in symbols:
            symbols.append(symbol)
    symbols = symbols[:_NEWS_SYMBOL_LIMIT]

    try:
        news_service.refresh(symbols, limit_per_symbol=per_symbol)
    except Exception as exc:
        logger.warning("News refresh failed during report build: %s", exc)

    seen: set[str] = set()
    articles: list[dict] = []
    for symbol in symbols:
        try:
            for article in news_service.get_recent(limit=per_symbol, symbol=symbol):
                url = article.get("url")
                if url and url not in seen:
                    seen.add(url)
                    articles.append(article)
        except Exception as exc:
            logger.warning("Could not read stored news for %s: %s", symbol, exc)

    articles.sort(
        key=lambda a: (a.get("published_at") is not None, a.get("published_at")),
        reverse=True,
    )
    return articles[:limit]


def format_headlines(articles: list[dict]) -> str:
    """Render collected headlines as a Markdown block."""
    if not articles:
        return ""
    lines = ["## Top news", ""]
    for article in articles:
        published = article.get("published_at")
        when = published.strftime("%Y-%m-%d %H:%M") if published else "undated"
        source = article.get("source") or "unknown source"
        symbols = ", ".join(article.get("symbols") or [])
        line = f"- [{article['headline']}]({article['url']}) — *{source}, {when}*"
        if symbols:
            line += f" [{symbols}]"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


class ReportPipeline:
    """Runs a prompt through interpret -> gather -> analyse -> write -> archive."""

    def __init__(self, recap_service, news_service=None) -> None:
        """
        Args:
            recap_service: A :class:`~market_intel.services.recap.RecapService`,
                which supplies both the snapshot and the model client.
            news_service: Optional :class:`~market_intel.services.news.NewsService`.
                When present the report carries the day's headlines; when absent
                everything else still works.
        """
        self._recap = recap_service
        self._news = news_service

    @property
    def ai_available(self) -> bool:
        """Whether stage 4 can use the model rather than the deterministic writer."""
        return self._recap.available

    def run(
        self,
        prompt: str,
        use_web_search: bool = True,
        now: dt.datetime | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        snapshot=None,
        include_news: bool = True,
    ) -> Report:
        """Produce a report for ``prompt``.

        Args:
            prompt: Free-text instruction, e.g. "brief wrap on APAC rates".
            use_web_search: Allow the model to research the news online.
            now: Clock override for session classification.
            progress: Optional ``(done, total, symbol)`` callback during pricing.
            snapshot: Reuse an already-built snapshot instead of pricing again.
                The scope of a reused snapshot is trusted as-is.
            include_news: Gather headlines for the movers. Best-effort - a
                news outage never fails the report.

        Returns:
            A :class:`Report`. Never raises for a missing API key - it falls
            back to the deterministic writer instead.
        """
        request = interpret_prompt(prompt, use_web_search=use_web_search)
        logger.info("Report request resolved: %s", request.scope_description)

        if snapshot is None:
            snapshot = self._recap.build_snapshot(
                regions=list(request.regions) or None, now=now, progress=progress
            )

        findings = compose_findings(snapshot)
        articles = gather_headlines(self._news, snapshot) if include_news else []
        stamp = snapshot.generated_at.strftime("%Y-%m-%d")
        title = _title_for(request, stamp)

        if self.ai_available:
            try:
                return self._write_with_model(
                    request, snapshot, findings, title, articles
                )
            except Exception as exc:
                # A model failure must not cost the user their report; fall back
                # to the deterministic writer and say so plainly.
                logger.warning("AI report generation failed, falling back: %s", exc)
                return self._write_deterministically(
                    request, snapshot, findings, title, articles,
                    note=f"AI generation failed: {exc}",
                )
        return self._write_deterministically(
            request, snapshot, findings, title, articles
        )

    def _write_with_model(self, request, snapshot, findings, title, articles) -> Report:
        """Stage 4a: hand the findings to Claude for research and prose."""
        focus_parts = [_DEPTH_GUIDANCE.get(request.depth, _DEPTH_GUIDANCE["standard"])]
        if request.prompt.strip():
            focus_parts.append(f"The user asked: {request.prompt.strip()}")
        if request.asset_classes:
            names = ", ".join(a.value for a in request.asset_classes)
            focus_parts.append(f"Give particular weight to: {names}.")
        focus_parts.append(
            "Pre-computed findings derived arithmetically from the snapshot "
            "follow. Treat them as established fact and build on them rather "
            "than recomputing:\n\n" + findings
        )
        if articles:
            # Giving the model the headlines we already hold saves it spending
            # its search budget rediscovering them, and lets it cite a specific
            # story against a specific move.
            focus_parts.append(
                "Headlines already collected for these instruments follow. Use "
                "them alongside your own search, and cite them where they "
                "explain a move:\n\n" + format_headlines(articles)
            )

        note = self._recap.generate_recap(
            snapshot,
            focus="\n\n".join(focus_parts),
            use_web_search=request.use_web_search,
            title=title,
            tags="report,recap",
        )
        return Report(
            title=title,
            content=note["content"],
            request=request,
            generated_by="claude",
            searches=note.get("searches", 0),
            instruments_priced=len(snapshot.moves),
            failures=dict(snapshot.failures),
            articles=articles,
            note_id=note["id"],
        )

    def _write_deterministically(
        self, request, snapshot, findings, title, articles, note: str | None = None
    ) -> Report:
        """Stage 4b: render the findings as prose, with no model involved."""
        header = [
            f"# {title}",
            "",
            f"*Scope: {request.scope_description} · "
            f"{len(snapshot.moves)} instruments priced*",
            "",
        ]
        if request.prompt.strip():
            header += [f"> **Your prompt:** {request.prompt.strip()}", ""]
        header += [
            "> Written without a language model: every statement below is "
            "computed directly from the snapshot. Set `MIP_ANTHROPIC_API_KEY` "
            "to add researched causation and the day's news.",
            "",
        ]
        if note:
            header += [f"> ⚠️ {note}", ""]

        body = header + [findings]
        if articles:
            body.append(format_headlines(articles))
        content = "\n".join(body)

        archived_id = None
        try:
            archived = self._recap._db and self._archive(title, content)
            archived_id = archived
        except Exception as exc:  # archiving is a convenience, not the product
            logger.warning("Could not archive deterministic report: %s", exc)

        return Report(
            title=title,
            content=content,
            request=request,
            generated_by="deterministic",
            instruments_priced=len(snapshot.moves),
            failures=dict(snapshot.failures),
            articles=articles,
            note_id=archived_id,
        )

    def _archive(self, title: str, content: str) -> int:
        """Store a deterministic report in the research archive."""
        from market_intel.database.repositories.research import ResearchRepository

        with self._recap._db.session() as session:
            note = ResearchRepository(session).add(
                title=title,
                content=content,
                note_type="market_report",
                model=None,
                tags="report,deterministic",
            )
            return note.id


def _title_for(request: ReportRequest, stamp: str) -> str:
    """Build an archive title reflecting the resolved scope."""
    if request.regions and len(request.regions) <= 2:
        scope = "/".join(r.value for r in request.regions)
        return f"{scope} market report - {stamp}"
    return f"Global market report - {stamp}"
