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
        fred_api_key: API key for FRED macroeconomic series.
        price_cache_ttl_minutes: How long cached price responses stay fresh.
        news_cache_ttl_minutes: How long cached news responses stay fresh.
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

    # Provider selection
    market_data_provider: str = "yfinance"
    news_provider: str = "yfinance"
    event_provider: str = "yfinance"
    etf_provider: str = "yfinance"

    # API keys (all optional; features degrade gracefully when absent)
    anthropic_api_key: str | None = None
    finnhub_api_key: str | None = None
    newsapi_api_key: str | None = None
    fred_api_key: str | None = None

    # Caching
    price_cache_ttl_minutes: int = 15
    news_cache_ttl_minutes: int = 30
    security_info_cache_ttl_minutes: int = 1440
    calendar_cache_ttl_minutes: int = 720
    etf_holdings_cache_ttl_minutes: int = 1440

    # Market data
    default_history_days: int = 365

    # AI research
    ai_model: str = "claude-opus-4-8"
    ai_max_tokens: int = 16000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()
