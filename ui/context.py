"""Shared application context for the Streamlit app.

Builds the full service graph once per server process (``st.cache_resource``)
so every page and rerun shares the same database engine, cache and providers.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from market_intel.cache import ApiCache
from market_intel.config import Settings, get_settings
from market_intel.database import Database
from market_intel.logging_conf import setup_logging
from market_intel.providers.registry import (
    create_etf_provider,
    create_event_provider,
    create_market_data_provider,
    create_news_provider,
)
from market_intel.services import MarketDataService, NewsService
from market_intel.services.calendar import CalendarService
from market_intel.services.etf import EtfService
from market_intel.services.portfolio import (
    JournalService,
    ThemeService,
    WatchlistService,
)
from market_intel.services.research import ResearchService
from market_intel.services.scanner import ScannerService


@dataclass(frozen=True)
class AppServices:
    """Bundle of everything a page needs."""

    settings: Settings
    db: Database
    market_data: MarketDataService
    news: NewsService
    calendar: CalendarService
    etf: EtfService
    scanner: ScannerService
    research: ResearchService
    watchlists: WatchlistService
    themes: ThemeService
    journal: JournalService


@st.cache_resource(show_spinner="Starting Market Intel...")
def get_services() -> AppServices:
    """Construct the service graph once per Streamlit server process."""
    settings = get_settings()
    setup_logging(settings)

    db = Database(settings.database_url)
    db.create_all()
    cache = ApiCache(db)
    cache.purge_expired()

    market_data = MarketDataService(
        db, create_market_data_provider(settings), cache, settings
    )
    return AppServices(
        settings=settings,
        db=db,
        market_data=market_data,
        news=NewsService(db, create_news_provider(settings), cache, settings, market_data),
        calendar=CalendarService(
            db, create_event_provider(settings), cache, settings, market_data
        ),
        etf=EtfService(db, create_etf_provider(settings), cache, settings, market_data),
        scanner=ScannerService(market_data),
        research=ResearchService(db, settings),
        watchlists=WatchlistService(db, market_data),
        themes=ThemeService(db, market_data),
        journal=JournalService(db, market_data),
    )
