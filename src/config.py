"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """Load .env file from project root if it exists."""
    env_path = _project_root() / ".env"
    load_dotenv(env_path)


def _get_camera_source() -> int | str:
    """Parse CAMERA_SOURCE from env. Integer means webcam index, string means file path."""
    raw = os.getenv("CAMERA_SOURCE", "0")
    try:
        return int(raw)
    except ValueError:
        return raw


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    All values are resolved once at construction time from environment
    variables (or their defaults).
    """

    camera_source: int | str = field(default_factory=_get_camera_source)
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.92"))
    )
    consecutive_hits: int = field(
        default_factory=lambda: int(os.getenv("CONSECUTIVE_HITS", "3"))
    )
    cooldown_seconds: int = field(
        default_factory=lambda: int(os.getenv("COOLDOWN_SECONDS", "5"))
    )
    clip_length: int = field(
        default_factory=lambda: int(os.getenv("CLIP_LENGTH", "16"))
    )
    dashboard_port: int = field(
        default_factory=lambda: int(os.getenv("DASHBOARD_PORT", "8000"))
    )
    model_path: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", "models/violence_detector.pt")
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "data/watchdog.db")
    )
    snapshot_dir: str = field(
        default_factory=lambda: os.getenv("SNAPSHOT_DIR", "data/snapshots")
    )
    log_dir: str = field(
        default_factory=lambda: os.getenv("LOG_DIR", "logs")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )


def get_settings() -> Settings:
    """Create a Settings instance after loading the .env file."""
    _load_env()
    return Settings()
