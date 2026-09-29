"""Application configuration.

All settings are read from environment variables prefixed with ``MIP_``
(e.g. ``MIP_DATABASE_URL``) or from a local ``.env`` file. No values are
hardcoded elsewhere in the codebase — modules receive a :class:`Settings`
instance (or call :func:`get_settings`) rather than reading the
environment directly, which keeps configuration testable and centralised.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings.

    Attributes:
        environment: Deployment environment name (``development`` / ``production``).
        database_url: SQLAlchemy connection URL. SQLite by default; point this
            at a PostgreSQL URL (``postgresql+psycopg://...``) to migrate.
        log_level: Root log level for the ``market_intel`` logger.
        log_dir: Directory for rotating log files (created if missing).
        market_data_provider: Name of the registered market-data adapter to use.
        news_provider: Name of the registered news adapter to use.
        anthropic_api_key: API key for AI-generated research summaries.
        finnhub_api_key: API key for fundamentals / earnings calendar.
        newsapi_api_key: API key for NewsAPI headlines.
        fred_api_key: API key for FRED macroeconomic series. Optional - the FRED
            adapter falls back to the keyless public CSV endpoint without one.
        alphavantage_api_key: API key for Alpha Vantage daily bars.
        alphavantage_daily_budget: Self-imposed request cap for Alpha Vantage,
            matching the free tier's 25/day allowance.
        price_cache_ttl_minutes: How long cached price responses stay fresh.
        news_cache_ttl_minutes: How long cached news responses stay fresh.
        recap_web_search_max_uses: Search budget for one agentic market recap.
            Each generation may run up to this many web searches while
            researching the day's news; higher means better sourced and slower.
        mkr_history_days: How much daily history the MKR analysis loads. Three
            years covers the 200-day average and gives the monthly timeframe
            enough bars to trend.
        mkr_horizon_days: Monte Carlo horizon for the MKR analysis.
        mkr_monte_carlo_paths: How many paths each Monte Carlo regime simulates.
        mkr_risk_free_rate: Annual risk-free rate used for Black-Scholes deltas
            and option repricing. It only feeds the greeks, so a slightly
            stale rate moves deltas in the third decimal, not the decision.
        mkr_web_search_max_uses: Search budget for one AI-written MKR analysis.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MIP_",
        extra="ignore",
    )

    # Core
    environment: str = "development"
    database_url: str = "sqlite:///data/market_intel.db"
    log_level: str = "INFO"
    log_dir: Path = Path("logs")

    # Provider selection.
    #
    # "chain" tries yfinance, then FRED, then Alpha Vantage, so one source
    # breaking degrades the data instead of emptying it. Set this to a single
    # adapter name ("yfinance", "fred", "alphavantage") to pin one source.
    market_data_provider: str = "chain"
    news_provider: str = "yfinance"
    event_provider: str = "yfinance"
    etf_provider: str = "yfinance"

    # API keys (all optional; features degrade gracefully when absent)
    anthropic_api_key: str | None = None
    finnhub_api_key: str | None = None
    newsapi_api_key: str | None = None
    fred_api_key: str | None = None
    alphavantage_api_key: str | None = None

    # Alpha Vantage's free tier allows 25 requests/day. The adapter refuses to
    # exceed this per process so a refresh cannot silently burn the allowance.
    alphavantage_daily_budget: int = 25

    # Caching
    price_cache_ttl_minutes: int = 15
    news_cache_ttl_minutes: int = 30
    security_info_cache_ttl_minutes: int = 1440
    calendar_cache_ttl_minutes: int = 720
    etf_holdings_cache_ttl_minutes: int = 1440
    fundamentals_cache_ttl_minutes: int = 360
    options_cache_ttl_minutes: int = 30

    # Market data
    default_history_days: int = 365

    # AI research
    ai_model: str = "claude-opus-5"
    ai_max_tokens: int = 16000

    # Global market recap
    recap_web_search_max_uses: int = 12

    # MKR 14-Framework single-name analysis
    mkr_history_days: int = 1100
    mkr_horizon_days: int = 90
    mkr_monte_carlo_paths: int = 1000
    mkr_risk_free_rate: float = 0.04
    mkr_web_search_max_uses: int = 8


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()
