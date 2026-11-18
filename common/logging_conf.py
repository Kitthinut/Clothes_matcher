"""Central logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED: set[str] = set()


def setup_logging(name: str = "raja_fashion", level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger with a consistent format, configuring it only once."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if name in _CONFIGURED:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED.add(name)
    return logger


LOGGER = setup_logging()
