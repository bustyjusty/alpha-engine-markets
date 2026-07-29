"""Tests for the news repository, service and yfinance parsing."""

import datetime as dt

import pytest

from market_intel.cache import ApiCache
from market_intel.config import Settings
from market_intel.database import Database
from market_intel.exceptions import ProviderError
from market_intel.models import NewsItem, PriceBar, SecurityInfo
from market_intel.providers.base import MarketDataProvider, NewsProvider
from market_intel.providers.yfinance_provider import parse_news_item
from market_intel.services import MarketDataService
from market_intel.services.news import NewsService

UTC = dt.timezone.utc


class FakeMarketProvider(MarketDataProvider):
    name = "fake"

    def get_security_info(self, symbol: str) -> SecurityInfo:
        return SecurityInfo(symbol=symbol)

    def get_daily_bars(self, symbol, start, end) -> list[PriceBar]:
        raise NotImplementedError

class FakeNewsProvider(NewsProvider):
    name = "fake"

    def __init__(self, items: dict[str, list[NewsItem]], fail: bool = False) -> None:
        self.items = items
        self.fail = fail
        self.calls: list[str] = []

    def get_news(self, symbol: str, limit: int = 20) -> list[NewsItem]:
        self.calls.append(symbol)
        if self.fail:
            raise ProviderError("down", provider=self.name)
        return self.items.get(symbol, [])


def _item(url: str, headline: str = "Headline", when: dt.datetime | None = None) -> NewsItem:
    return NewsItem(
        headline=headline,
        url=url,
        source="TestWire",
        published_at=when or dt.datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
    )


@pytest.fixture()
def db() -> Database:
    database = Database("sqlite:///:memory:")
    database.create_all()
    return database


def _news_service(db: Database, provider: FakeNewsProvider) -> NewsService:
    settings = Settings(_env_file=None)
    cache = ApiCache(db)
    market_data = MarketDataService(db, FakeMarketProvider(), cache, settings)
    return NewsService(db, provider, cache, settings, market_data)


class TestNewsService:
    def test_refresh_stores_and_links(self, db: Database) -> None:
        provider = FakeNewsProvider({"NVDA": [_item("https://x/1"), _item("https://x/2")]})
        service = _news_service(db, provider)

        assert service.refresh(["NVDA"]) == 2
        recent = service.get_recent()
        assert len(recent) == 2
        assert recent[0]["symbols"] == ["NVDA"]

    def test_same_url_across_symbols_deduplicates(self, db: Database) -> None:
        shared = _item("https://x/shared")
        provider = FakeNewsProvider({"NVDA": [shared], "AMD": [shared]})
        service = _news_service(db, provider)

        service.refresh(["NVDA", "AMD"])
        recent = service.get_recent()
        assert len(recent) == 1
        assert recent[0]["symbols"] == ["AMD", "NVDA"]

    def test_refresh_throttled_by_ttl_marker(self, db: Database) -> None:
        provider = FakeNewsProvider({"NVDA": [_item("https://x/1")]})
        service = _news_service(db, provider)

        service.refresh(["NVDA"])
        service.refresh(["NVDA"])
        assert provider.calls == ["NVDA"]

    def test_provider_failure_skips_symbol_quietly(self, db: Database) -> None:
        service = _news_service(db, FakeNewsProvider({}, fail=True))
        assert service.refresh(["NVDA"]) == 0

    def test_get_recent_filters_by_symbol(self, db: Database) -> None:
        provider = FakeNewsProvider(
            {
                "NVDA": [_item("https://x/nvda")],
                "AMD": [_item("https://x/amd")],
            }
        )
        service = _news_service(db, provider)
        service.refresh(["NVDA", "AMD"])

        nvda_news = service.get_recent(symbol="NVDA")
        assert len(nvda_news) == 1
        assert nvda_news[0]["url"] == "https://x/nvda"
        assert service.get_recent(symbol="ZZZZ") == []

    def test_get_recent_orders_newest_first(self, db: Database) -> None:
        provider = FakeNewsProvider(
            {
                "NVDA": [
                    _item("https://x/old", when=dt.datetime(2026, 7, 1, tzinfo=UTC)),
                    _item("https://x/new", when=dt.datetime(2026, 7, 10, tzinfo=UTC)),
                ]
            }
        )
        service = _news_service(db, provider)
        service.refresh(["NVDA"])
        urls = [a["url"] for a in service.get_recent()]
        assert urls == ["https://x/new", "https://x/old"]


class TestParseNewsItem:
    def test_current_nested_format(self) -> None:
        raw = {
            "content": {
                "title": "Chips rally",
                "summary": "Semis up.",
                "pubDate": "2026-07-10T12:30:00Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://example.com/a"},
            }
        }
        item = parse_news_item(raw, symbol="NVDA")
        assert item is not None
        assert item.headline == "Chips rally"
        assert item.url == "https://example.com/a"
        assert item.source == "Reuters"
        assert item.published_at == dt.datetime(2026, 7, 10, 12, 30, tzinfo=UTC)
        assert item.symbols == ("NVDA",)

    def test_legacy_flat_format(self) -> None:
        raw = {
            "title": "Old style",
            "link": "https://example.com/b",
            "publisher": "Bloomberg",
            "providerPublishTime": 1783168200,
        }
        item = parse_news_item(raw, symbol="AMD")
        assert item is not None
        assert item.url == "https://example.com/b"
        assert item.source == "Bloomberg"
        assert item.published_at is not None

    def test_missing_url_returns_none(self) -> None:
        assert parse_news_item({"content": {"title": "No url"}}, symbol="X") is None
