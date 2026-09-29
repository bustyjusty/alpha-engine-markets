"""Market-data service: fetch, persist and serve price history.

Responsibilities:
    * Keep the local database the source of truth the UI reads from.
    * Fetch only missing date ranges from the provider (incremental sync).
    * Rate-limit provider hits with a cache marker so dashboard reruns
      within the TTL never touch the network.
    * Degrade gracefully: if the provider fails but stored data exists,
      serve the stored data and log a warning instead of raising.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database, orm
from market_intel.database.repositories import PriceRepository, SecurityRepository
from market_intel.exceptions import DataNotFoundError, ProviderError
from market_intel.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

_FRAME_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "source"]

#: How many trailing days to re-fetch on a sync, so the current session's bar
#: keeps updating and provider revisions to recent bars are picked up.
_REFRESH_TAIL_DAYS = 3


class MarketDataService:
    """High-level access to securities and their price history."""

    def __init__(
        self,
        db: Database,
        provider: MarketDataProvider,
        cache: ApiCache,
        settings: Settings,
    ) -> None:
        self._db = db
        self._provider = provider
        self._cache = cache
        self._settings = settings

    # --- Securities ---------------------------------------------------------

    def ensure_security(self, symbol: str) -> int:
        """Return the security's DB id, creating or enriching it if needed.

        Reference data is refreshed from the provider at most once per
        ``security_info_cache_ttl_minutes``. If the provider is down but a
        stored record exists, the stored record is used.

        Raises:
            ProviderError: The security is unknown locally and the provider
                could not supply it.
        """
        symbol = symbol.strip().upper()
        cache_key = f"security_info:{self._provider.name}:{symbol}"

        with self._db.session() as session:
            existing = SecurityRepository(session).get_by_symbol(symbol)

        if existing is not None and self._cache.get(cache_key) is not None:
            return existing.id

        try:
            info = self._provider.get_security_info(symbol)
        except ProviderError as exc:
            if existing is not None:
                logger.warning(
                    "Could not refresh info for %s (%s); using stored record",
                    symbol,
                    exc,
                )
                return existing.id
            raise

        with self._db.session() as session:
            security_id = SecurityRepository(session).upsert(info).id
        self._cache.set(
            cache_key,
            {"refreshed_at": dt.datetime.now(dt.timezone.utc).isoformat()},
            ttl_minutes=self._settings.security_info_cache_ttl_minutes,
        )
        return security_id

    # --- Price history --------------------------------------------------------

    def get_price_history(
        self,
        symbol: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        """Return daily bars as a DataFrame indexed by date.

        Missing data is fetched from the provider and persisted first, so
        repeated calls are served from the local database.

        Args:
            symbol: Ticker symbol (case-insensitive).
            start: First date wanted; defaults to ``default_history_days`` ago.
            end: Last date wanted; defaults to today.
        """
        end = end or dt.date.today()
        start = start or end - dt.timedelta(days=self._settings.default_history_days)
        security_id = self.ensure_security(symbol)

        self._sync_range(symbol, security_id, start, end)

        with self._db.session() as session:
            bars = PriceRepository(session).get_history(security_id, start, end)
        return _bars_to_frame(bars)

    def _sync_range(
        self, symbol: str, security_id: int, start: dt.date, end: dt.date
    ) -> None:
        """Fetch and persist whatever part of [start, end] is missing locally."""
        marker_key = f"prices_synced:{self._provider.name}:{symbol}"
        marker = self._cache.get(marker_key)
        if (
            marker is not None
            and dt.date.fromisoformat(marker["start"]) <= start
            and dt.date.fromisoformat(marker["end"]) >= end
        ):
            return  # synced this range within the TTL; skip the provider

        with self._db.session() as session:
            repo = PriceRepository(session)
            earliest = repo.earliest_date(security_id)
            latest = repo.latest_date(security_id)

        windows: list[tuple[dt.date, dt.date]] = []
        if latest is None:
            windows.append((start, end))
        else:
            if earliest is not None and start < earliest:
                windows.append((start, earliest - dt.timedelta(days=1)))

            # Fetching only *missing* dates leaves stored bars frozen at their
            # first print: today's bar would never update intraday, and a bad
            # bar the provider later corrects (a zero-volume print with a wildly
            # wrong close) would persist forever, since no gap remains to fill.
            #
            # So always re-fetch from whichever is earlier: the start of the gap,
            # or a short trailing window. That covers new days and revisions to
            # recent ones in a single request. Upserts make it idempotent.
            gap_start = latest + dt.timedelta(days=1)
            tail_start = end - dt.timedelta(days=_REFRESH_TAIL_DAYS)
            window_start = max(start, min(gap_start, tail_start))
            if window_start <= end:
                windows.append((window_start, end))

        for window_start, window_end in windows:
            try:
                bars = self._provider.get_daily_bars(symbol, window_start, window_end)
            except DataNotFoundError:
                # Normal for weekends/holidays or pre-listing dates.
                logger.debug(
                    "No bars for %s in %s..%s", symbol, window_start, window_end
                )
                continue
            except ProviderError as exc:
                if latest is None:
                    raise
                logger.warning(
                    "Provider %s failed for %s (%s); serving stored data",
                    self._provider.name,
                    symbol,
                    exc,
                )
                continue
            with self._db.session() as session:
                PriceRepository(session).upsert_bars(security_id, bars)
            logger.info(
                "Stored %d bars for %s (%s..%s)",
                len(bars),
                symbol,
                window_start,
                window_end,
            )

        self._cache.set(
            marker_key,
            {"start": start.isoformat(), "end": end.isoformat()},
            ttl_minutes=self._settings.price_cache_ttl_minutes,
        )


def _bars_to_frame(bars: list[orm.PriceBar]) -> pd.DataFrame:
    """Convert stored bars to a date-indexed DataFrame."""
    if not bars:
        return pd.DataFrame(columns=_FRAME_COLUMNS, index=pd.Index([], name="date"))
    frame = pd.DataFrame(
        [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "adj_close": bar.adj_close,
                "volume": bar.volume,
                "source": bar.source,
            }
            for bar in bars
        ]
    )
    return frame.set_index("date")
