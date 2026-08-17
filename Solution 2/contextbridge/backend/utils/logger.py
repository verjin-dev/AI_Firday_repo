"""Loguru-backed logger with a stdlib fallback so imports never explode."""

from __future__ import annotations

import sys

try:
    from loguru import logger as _logger

    _logger.remove()
    _logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
    )
    logger = _logger
except ImportError:  # pragma: no cover - loguru is a declared dependency
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    )
    logger = logging.getLogger("contextbridge")  # type: ignore[assignment]


def configure(level: str = "INFO") -> None:
    """Re-point the sink at a different level (called from main.py at startup)."""
    try:
        logger.remove()
        logger.add(
            sys.stderr,
            level=level.upper(),
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan> - <level>{message}</level>"
            ),
        )
    except AttributeError:  # stdlib fallback
        pass


__all__ = ["logger", "configure"]
