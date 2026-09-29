"""Global market recap service: cross-asset snapshot plus an AI-written wrap.

Two halves that are deliberately independent:

* :meth:`RecapService.build_snapshot` is the **quantitative** half. It walks the
  instrument universe through :class:`~market_intel.services.market_data.MarketDataService`,
  computes every move, and returns a structured snapshot. It needs no API key and
  degrades per-instrument: one dead ticker is recorded as a failure, not an
  exception that loses the other eighty.

* :meth:`RecapService.generate_recap` is the **narrative** half. It hands the
  snapshot to Claude with the server-side web-search tool enabled, so the model
  researches the day's actual headlines and explains the numbers it was given.

The division matters for trust. The model is told, in the system prompt and
again in the payload, that every level and move is supplied and must be quoted
verbatim - web search is for *causation and news*, never for prices. A trader
reading the wrap should never have to wonder whether a number was invented.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import anthropic

from market_intel.analysis.recap import (
    BlockStats,
    InstrumentMove,
    SessionState,
    block_stats,
    compute_move,
    group_moves,
    rank_by_magnitude,
    rank_rates_by_bp,
    session_label,
    session_state,
    top_gainers,
    top_losers,
)
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories.research import ResearchRepository
from market_intel.exceptions import (
    ConfigurationError,
    MarketIntelError,
    ProviderError,
    RateLimitError,
)
from market_intel.services.market_data import MarketDataService
from market_intel.universe import (
    ASSET_CLASS_ORDER,
    REGION_ORDER,
    AssetClass,
    Instrument,
    Region,
    instruments_for,
)

logger = logging.getLogger(__name__)

#: Days of history pulled per instrument. Long enough that year-to-date and
#: one-month windows are always available, including in early January.
_HISTORY_DAYS = 400

_SYSTEM_PROMPT = (
    "You are a senior cross-asset strategist writing the morning market wrap for "
    "a professional trading desk. Your reader is a trader who needs to form a view "
    "in ninety seconds.\n\n"
    "ABSOLUTE RULE ON NUMBERS: every price, level, yield and percentage move is "
    "supplied to you in the MARKET SNAPSHOT below. Quote those figures exactly as "
    "given. Never state a level or a move that is not in the snapshot, and never "
    "adjust one you were given. If you want to cite a number that is not there, "
    "either search for it and attribute the source, or leave it out.\n\n"
    "Use the web search tool to establish WHY things moved: the day's headlines, "
    "data releases, central-bank commentary, earnings and geopolitics. The "
    "snapshot tells you what happened; searching tells you why.\n\n"
    "Structure the wrap by region, and within each region by asset class "
    "(Equities, Fixed Income, Currencies, Commodities). Respect the session "
    "status given for each region - if a region is closed, say you are describing "
    "its last completed session rather than implying it traded today.\n\n"
    "Write like a desk note, not a press release: lead with what changed and what "
    "it implies, name the biggest movers, flag divergences between asset classes, "
    "and close with what to watch next. Separate fact from interpretation. This is "
    "decision-support research, not financial advice - do not give trade "
    "instructions or price targets."
)


@dataclass(frozen=True, slots=True)
class RegionBlock:
    """One region's slice of the snapshot.

    Attributes:
        region: The region described.
        state: Whether its cash market is open right now.
        state_label: Human description of what the numbers represent.
        by_asset_class: Moves grouped by asset class, in display order.
        stats: Breadth summary across every instrument in the region.
    """

    region: Region
    state: SessionState
    state_label: str
    by_asset_class: dict[AssetClass, list[InstrumentMove]]
    stats: BlockStats

    @property
    def moves(self) -> list[InstrumentMove]:
        """Every move in the region, flattened in display order."""
        return [move for block in self.by_asset_class.values() for move in block]


@dataclass(frozen=True, slots=True)
class RecapSnapshot:
    """The full cross-asset picture at a point in time.

    Attributes:
        generated_at: When the snapshot was built (UTC).
        blocks: Region blocks in session-clock order.
        failures: ``symbol -> reason`` for instruments that could not be priced.
    """

    generated_at: dt.datetime
    blocks: list[RegionBlock]
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def moves(self) -> list[InstrumentMove]:
        """Every computed move across every region."""
        return [move for block in self.blocks for move in block.moves]

    @property
    def is_empty(self) -> bool:
        """True when nothing could be priced at all."""
        return not self.moves

    def region(self, region: Region) -> RegionBlock | None:
        """Return one region's block, or None if it has no data."""
        return next((block for block in self.blocks if block.region is region), None)

    def biggest_movers(self, limit: int = 8) -> list[InstrumentMove]:
        """The largest absolute price moves across the whole snapshot.

        Rates are excluded here and ranked separately by
        :meth:`biggest_rate_moves`, since basis points and percent do not
        compare.
        """
        return rank_by_magnitude(self.moves, limit)

    def biggest_rate_moves(self, limit: int = 6) -> list[InstrumentMove]:
        """The largest yield and spread moves, in basis points."""
        return rank_rates_by_bp(self.moves, limit)


class RecapService:
    """Builds the cross-asset snapshot and the AI-written market wrap."""

    def __init__(
        self,
        db: Database,
        market_data: MarketDataService,
        settings: Settings,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._db = db
        self._market_data = market_data
        self._settings = settings
        self._client = client  # injectable for tests

    @property
    def available(self) -> bool:
        """Whether AI wrap generation can run (key configured or client injected)."""
        return self._client is not None or bool(self._settings.anthropic_api_key)

    # --- Quantitative half ----------------------------------------------------

    def build_snapshot(
        self,
        regions: Sequence[Region] | None = None,
        now: dt.datetime | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> RecapSnapshot:
        """Price the universe and compute every move.

        Instruments are fetched one at a time through the market-data service, so
        results are persisted and later runs are served from the local database.
        A failure on one symbol is recorded and skipped.

        Args:
            regions: Restrict to these regions; ``None`` covers the whole universe.
            now: Clock override for session classification (defaults to UTC now).
            progress: Optional ``(done, total, symbol)`` callback for UI feedback.

        Returns:
            A :class:`RecapSnapshot`, possibly with an empty ``blocks`` list if
            every instrument failed.
        """
        now = now or dt.datetime.now(dt.timezone.utc)
        wanted = list(regions) if regions else list(REGION_ORDER)
        universe: list[Instrument] = [
            instrument
            for region in wanted
            for instrument in instruments_for(region)
        ]

        start = dt.date.today() - dt.timedelta(days=_HISTORY_DAYS)
        moves: list[InstrumentMove] = []
        failures: dict[str, str] = {}

        for index, instrument in enumerate(universe, start=1):
            if progress is not None:
                progress(index, len(universe), instrument.symbol)
            try:
                frame = self._market_data.get_price_history(
                    instrument.symbol, start=start
                )
            except MarketIntelError as exc:
                failures[instrument.symbol] = str(exc)
                continue
            except Exception as exc:  # provider bugs must not kill the recap
                logger.warning("Unexpected error pricing %s: %s", instrument.symbol, exc)
                failures[instrument.symbol] = f"unexpected error: {exc}"
                continue

            if frame.empty:
                failures[instrument.symbol] = "no price history returned"
                continue

            closes = frame["adj_close"].fillna(frame["close"])
            volumes = frame["volume"] if "volume" in frame.columns else None
            move = compute_move(instrument, closes, volumes)
            if move is None:
                failures[instrument.symbol] = "insufficient history for a move"
                continue
            moves.append(move)

        grouped = group_moves(moves)
        blocks: list[RegionBlock] = []
        for region in REGION_ORDER:
            if region not in grouped or region not in wanted:
                continue
            ordered = {
                asset_class: grouped[region][asset_class]
                for asset_class in ASSET_CLASS_ORDER
                if asset_class in grouped[region]
            }
            state = session_state(region, now)
            blocks.append(
                RegionBlock(
                    region=region,
                    state=state,
                    state_label=session_label(region, state),
                    by_asset_class=ordered,
                    stats=block_stats(
                        [move for block in ordered.values() for move in block]
                    ),
                )
            )

        logger.info(
            "Recap snapshot: %d instruments priced, %d failures",
            len(moves),
            len(failures),
        )
        return RecapSnapshot(generated_at=now, blocks=blocks, failures=failures)

    # --- Narrative half -------------------------------------------------------

    def generate_recap(
        self,
        snapshot: RecapSnapshot,
        focus: str | None = None,
        use_web_search: bool = True,
        title: str | None = None,
        tags: str | None = "recap",
    ) -> dict[str, Any]:
        """Write the market wrap from a snapshot and archive it.

        Args:
            snapshot: The quantitative picture the wrap must describe.
            focus: Optional extra steer (e.g. "focus on rates and the yen").
            use_web_search: Let the model research the day's news online. Turning
                this off produces a numbers-only wrap with no causation.
            title: Archive title; defaults to a dated "Market recap" heading.
            tags: Comma-separated archive tags.

        Returns:
            The archived note as a dict, with an extra ``searches`` key counting
            how many web searches the model ran.

        Raises:
            ConfigurationError: No API key configured.
            RateLimitError: The API rate limit was hit.
            ProviderError: Any other API failure, refusal, or empty response.
        """
        client = self._get_client()
        payload = format_snapshot(snapshot)
        instruction = (
            "Write today's cross-asset market recap from the snapshot below.\n\n"
            "Cover every region present, and within each one every asset class "
            "present. Open with a short 'what matters today' summary, then the "
            "regional sections, then the biggest movers across all regions, then "
            "what to watch next. Use the web search tool to find the news and "
            "data releases that explain these moves, and attribute what you find."
        )
        if focus:
            instruction += f"\n\nExtra focus for this wrap: {focus}"

        tools: list[dict[str, Any]] = []
        if use_web_search:
            tools.append(
                {
                    "type": "web_search_20260318",
                    "name": "web_search",
                    "max_uses": self._settings.recap_web_search_max_uses,
                }
            )

        try:
            response = client.messages.create(
                model=self._settings.ai_model,
                max_tokens=self._settings.ai_max_tokens,
                system=_SYSTEM_PROMPT,
                tools=tools or anthropic.NOT_GIVEN,
                messages=[{"role": "user", "content": f"{instruction}\n\n{payload}"}],
            )
        except anthropic.RateLimitError as exc:
            raise RateLimitError(
                f"Anthropic rate limit hit: {exc}", provider="anthropic"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error ({exc.status_code}): {exc.message}",
                provider="anthropic",
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(
                f"Could not reach the Anthropic API: {exc}", provider="anthropic"
            ) from exc

        if response.stop_reason == "refusal":
            raise ProviderError(
                "The model declined to write this recap", provider="anthropic"
            )
        if response.stop_reason == "max_tokens":
            logger.warning("Recap hit the max_tokens limit and may be truncated")

        content = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not content:
            raise ProviderError("Model returned an empty recap", provider="anthropic")

        searches = sum(
            1 for block in response.content if block.type == "server_tool_use"
        )
        stamp = snapshot.generated_at.strftime("%Y-%m-%d")
        with self._db.session() as session:
            note = ResearchRepository(session).add(
                title=title or f"Market recap - {stamp}",
                content=content,
                note_type="market_recap",
                model=response.model,
                tags=tags,
            )
            archived = {
                "id": note.id,
                "title": note.title,
                "content": note.content,
                "note_type": note.note_type,
                "model": note.model,
                "tags": note.tags,
                "created_at": note.created_at,
            }
        archived["searches"] = searches
        logger.info("Recap generated (%d web searches)", searches)
        return archived

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise ConfigurationError(
                    "MIP_ANTHROPIC_API_KEY is not set - AI recap generation is disabled"
                )
            self._client = anthropic.Anthropic(api_key=self._settings.anthropic_api_key)
        return self._client


def format_snapshot(snapshot: RecapSnapshot) -> str:
    """Render a snapshot as the Markdown block handed to the model.

    Every level and move the wrap is allowed to quote appears here, which is what
    makes the "never invent a number" instruction enforceable.
    """
    stamp = snapshot.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# MARKET SNAPSHOT",
        f"Generated: {stamp}",
        "",
        "These are the only figures you may quote. Session status per region "
        "tells you whether a block is a live print or a prior close.",
        "",
    ]

    for block in snapshot.blocks:
        lines.append(f"## {block.region.value} - {block.state_label}")
        lines.append(
            f"Breadth: {block.stats.advancers} up / {block.stats.decliners} down "
            f"of {block.stats.count}, average move {block.stats.average_pct:+.2f}%"
        )
        lines.append("")
        for asset_class, moves in block.by_asset_class.items():
            lines.append(f"### {asset_class.value}")
            for move in moves:
                lines.append(_format_move_line(move))
            lines.append("")

    movers = snapshot.biggest_movers(10)
    if movers:
        lines.append("## Biggest price moves across all regions")
        for move in movers:
            lines.append(
                f"- {move.instrument.label} ({move.instrument.region.value}, "
                f"{move.instrument.asset_class.value}): {move.display_change}"
            )
        lines.append("")

    rate_moves = snapshot.biggest_rate_moves(8)
    if rate_moves:
        lines.append("## Biggest rate and spread moves (basis points)")
        lines.append(
            "Ranked separately from the list above: basis points and percent "
            "are different units and must not be compared directly."
        )
        for move in rate_moves:
            lines.append(
                f"- {move.instrument.label} ({move.instrument.region.value}): "
                f"{move.display_change}"
            )
        lines.append("")

    gainers = top_gainers(snapshot.moves, 5)
    losers = top_losers(snapshot.moves, 5)
    if gainers:
        lines.append("## Top gainers")
        lines += [f"- {m.instrument.label}: {m.display_change}" for m in gainers]
        lines.append("")
    if losers:
        lines.append("## Top losers")
        lines += [f"- {m.instrument.label}: {m.display_change}" for m in losers]
        lines.append("")

    if snapshot.failures:
        lines.append("## Instruments with no data (do not mention as moves)")
        lines += [f"- {symbol}: {reason}" for symbol, reason in snapshot.failures.items()]
        lines.append("")

    return "\n".join(lines)


def _format_move_line(move: InstrumentMove) -> str:
    """Render one instrument as a single prompt line."""
    parts = [
        f"- {move.instrument.label} [{move.instrument.symbol}]:",
        f"last {move.last:,.4g}",
        f"| day {move.display_change}",
    ]
    if move.ret_5d is not None:
        parts.append(f"| 5d {move.ret_5d:+.2f}%")
    if move.ret_1m is not None:
        parts.append(f"| 1m {move.ret_1m:+.2f}%")
    if move.ret_ytd is not None:
        parts.append(f"| ytd {move.ret_ytd:+.2f}%")
    parts.append(f"| as of {move.as_of:%Y-%m-%d}")
    if move.instrument.is_proxy:
        parts.append(f"| PROXY: {move.instrument.note}")
    elif move.instrument.note:
        parts.append(f"| {move.instrument.note}")
    return " ".join(parts)
