"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory that is cleaned up after the test."""
    return tmp_path
