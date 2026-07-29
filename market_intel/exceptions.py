"""Typed exception hierarchy for the platform.

Services catch :class:`ProviderError` subclasses at the boundary and
degrade gracefully (log, fall back to cached data, surface a friendly
message in the UI) instead of letting raw HTTP errors propagate.
"""

from __future__ import annotations


class MarketIntelError(Exception):
    """Base class for all platform-specific errors."""


class ConfigurationError(MarketIntelError):
    """A required setting or API key is missing or invalid."""


class ProviderError(MarketIntelError):
    """An external data provider failed.

    Attributes:
        provider: Name of the provider that raised the error.
    """

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class RateLimitError(ProviderError):
    """The provider rejected the request due to rate limiting."""


class DataNotFoundError(ProviderError):
    """The provider returned no data for the requested symbol/range."""
