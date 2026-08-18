"""loguru configuration: human sink on stderr, JSON sink on ./logs/app.jsonl."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_configured = False


def setup_logging(level: str = "INFO") -> "logger.__class__":
    global _configured
    if _configured:
        return logger
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "app.jsonl", level=level, serialize=True, rotation="10 MB", retention=3)
    _configured = True
    return logger


log = setup_logging()
