"""Structured logging for Cricket DRS."""

from __future__ import annotations

import sys
from functools import lru_cache

from config.settings import settings

try:
    from loguru import logger as _logger
except Exception:  # pragma: no cover
    import logging

    _logger = None
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))


def configure_logging() -> None:
    if _logger is None:
        return
    log_dir = settings.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    )
    _logger.add(
        log_dir / "drs_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )


@lru_cache(maxsize=1)
def _configured() -> bool:
    configure_logging()
    return True


def get_logger(name: str, level: int | None = None):
    _configured()
    if _logger is None:
        import logging

        return logging.getLogger(name)
    return _logger.bind(name=name)
