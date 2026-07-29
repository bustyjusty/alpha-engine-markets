"""Centralised logging configuration.

Every module obtains its logger with ``logging.getLogger(__name__)``.
Because all modules live under the ``market_intel`` package, they inherit
handlers from the single ``market_intel`` logger configured here.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from market_intel.config import Settings

_LOGGER_NAME = "market_intel"
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure console + rotating-file logging for the platform.

    Idempotent: calling this more than once (e.g. on Streamlit reruns)
    does not attach duplicate handlers.

    Args:
        settings: Application settings providing log level and directory.

    Returns:
        The configured package-level logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level.upper())
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_dir / "market_intel.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
