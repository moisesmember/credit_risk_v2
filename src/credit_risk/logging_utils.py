from __future__ import annotations

import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    resolved_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(resolved_level)
        return

    logging.basicConfig(level=resolved_level, format=DEFAULT_LOG_FORMAT)
