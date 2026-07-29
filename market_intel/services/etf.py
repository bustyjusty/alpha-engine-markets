"""ETF service: constituent snapshots and exposure mapping."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.database.repositories.etf import EtfHoldingRepository
from market_intel.exceptions import MarketIntelError, ProviderError
from market_intel.providers.base import EtfHoldingsProvider
from market_intel.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class EtfService:
    """Tracks ETF constituents and answers "who holds this stock?"."""

    def __init__(
        self,
        db: Database,
        provider: EtfHoldingsProvider,
        cache: ApiCache,
        settings: Settings,
        market_data: MarketDataService,
    ) -> None:
        self._db = db
        self._provider = provider
        self._cache = cache
        self._settings = settings
        self._market_data = market_data

    def refresh_holdings(self, etf_symbol: str) -> int:
        """Fetch and store today's holdings snapshot (TTL-throttled).

        Returns:
            Number of holdings stored (0 if throttled or provider failed
            with an existing snapshot present).
        """
        symbol = etf_symbol.strip().upper()
        marker_key = f"etf_holdings_synced:{self._provider.name}:{symbol}"
        if self._cache.get(marker_key) is not None:
            return 0

        security_id = self._market_data.ensure_security(symbol)
        try:
            items = self._provider.get_etf_holdings(symbol)
        except ProviderError as exc:
            with self._db.session() as session:
                has_snapshot = bool(
                    EtfHoldingRepository(session).latest_snapshot(security_id)
                )
            if has_snapshot:
                logger.warning(
                    "Holdings refresh failed for %s (%s); keeping stored snapshot",
                    symbol,
                    exc,
                )
                return 0
            raise

        with self._db.session() as session:
            count = EtfHoldingRepository(session).replace_snapshot(
                security_id,
                items,
                as_of=dt.date.today(),
                source=self._provider.name,
            )
        self._cache.set(
            marker_key,
            {"refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat()},
            ttl_minutes=self._settings.etf_holdings_cache_ttl_minutes,
        )
        logger.info("Stored %d holdings for %s", count, symbol)
        return count

    def get_holdings(self, etf_symbol: str) -> pd.DataFrame:
        """Latest stored snapshot as a DataFrame (symbol, name, weight, as_of)."""
        security_id = self._market_data.ensure_security(etf_symbol)
        with self._db.session() as session:
            rows = EtfHoldingRepository(session).latest_snapshot(security_id)
            return pd.DataFrame(
                [
                    {
                        "symbol": row.holding_symbol,
                        "name": row.holding_name,
                        "weight": row.weight,
                        "as_of": row.as_of,
                    }
                    for row in rows
                ]
            )

    def refresh_many(self, etf_symbols: Sequence[str]) -> dict[str, str]:
        """Refresh several ETFs, reporting per-symbol outcomes.

        Returns:
            Mapping of symbol -> "fetched" | "cached" | error message.
            Never raises for individual failures, so one bad symbol does
            not abort a batch.
        """
        outcomes: dict[str, str] = {}
        for raw in etf_symbols:
            symbol = raw.strip().upper()
            try:
                count = self.refresh_holdings(symbol)
                outcomes[symbol] = "fetched" if count else "cached"
            except MarketIntelError as exc:
                logger.warning("Batch holdings refresh failed for %s: %s", symbol, exc)
                outcomes[symbol] = str(exc)
        return outcomes

    def get_overlap(self, etf_a: str, etf_b: str) -> pd.DataFrame:
        """Common holdings between two ETFs' latest stored snapshots.

        Returns:
            DataFrame with symbol, name, weight_a, weight_b and
            overlap_weight (the smaller of the two weights — the standard
            pairwise overlap contribution), sorted by overlap descending.
            Empty if either ETF has no stored snapshot.
        """
        left = self.get_holdings(etf_a)
        right = self.get_holdings(etf_b)
        if left.empty or right.empty:
            return pd.DataFrame()
        merged = left.merge(
            right, on="symbol", how="inner", suffixes=("_a", "_b")
        )
        if merged.empty:
            return pd.DataFrame()
        merged["name"] = merged["name_a"].fillna(merged["name_b"])
        weights_a = pd.to_numeric(merged["weight_a"], errors="coerce")
        weights_b = pd.to_numeric(merged["weight_b"], errors="coerce")
        merged["overlap_weight"] = pd.concat([weights_a, weights_b], axis=1).min(axis=1)
        result = merged[["symbol", "name", "weight_a", "weight_b", "overlap_weight"]]
        return result.sort_values("overlap_weight", ascending=False).reset_index(drop=True)

    def get_exposure(self, symbol: str) -> list[dict[str, Any]]:
        """Which tracked ETFs hold this symbol, and at what weight."""
        with self._db.session() as session:
            pairs = EtfHoldingRepository(session).exposure_for_symbol(symbol)
            return [
                {
                    "etf": etf.symbol,
                    "etf_name": etf.name,
                    "weight": holding.weight,
                    "as_of": holding.as_of,
                }
                for etf, holding in pairs
            ]
