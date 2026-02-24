"""
logging.py — Structured logging setup for the Omni service.
"""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a consistent format.
    Safe to call multiple times (basicConfig is a no-op after first call).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
