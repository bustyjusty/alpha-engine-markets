"""Tests for application configuration."""

from pathlib import Path

import pytest

from market_intel.config import Settings


@pytest.fixture()
def clean_settings(monkeypatch: pytest.MonkeyPatch):
    """Build Settings ignoring any real .env file and stray MIP_ env vars."""

    def _build(**env: str) -> Settings:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return Settings(_env_file=None)

    return _build


def test_defaults(clean_settings) -> None:
    settings = clean_settings()
    assert settings.database_url == "sqlite:///data/market_intel.db"
    assert settings.market_data_provider == "yfinance"
    assert settings.anthropic_api_key is None
    assert settings.log_dir == Path("logs")


def test_environment_override(clean_settings) -> None:
    settings = clean_settings(
        MIP_DATABASE_URL="postgresql+psycopg://user:pw@localhost/mi",
        MIP_LOG_LEVEL="DEBUG",
        MIP_FINNHUB_API_KEY="test-key",
    )
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.log_level == "DEBUG"
    assert settings.finnhub_api_key == "test-key"


def test_unknown_env_vars_ignored(clean_settings) -> None:
    settings = clean_settings(MIP_NOT_A_REAL_SETTING="whatever")
    assert not hasattr(settings, "not_a_real_setting")
