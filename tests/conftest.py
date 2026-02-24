"""
conftest.py — Pytest shared fixtures for v2 tests.

Ensures the v2 root is on sys.path so 'from app...' works
regardless of where pytest is invoked from.
Also resets the settings singleton between tests that patch env vars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# v2/ root must be on the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reset_settings_singleton():
    """
    Reset the Settings singleton before every test.
    Ensures env-var patches (monkeypatch.setenv) take effect cleanly.
    """
    from app.core.config import reset_settings
    reset_settings()
    yield
    reset_settings()
