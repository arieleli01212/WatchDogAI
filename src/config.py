"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
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


def _parse_source(raw: str) -> int | str:
    """Parse a camera source. Integer means webcam index, string means file path or stream URL."""
    try:
        return int(raw)
    except ValueError:
        return raw


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single camera.

    Attributes
    ----------
    id:
        Unique identifier used in alerts, clips, and dashboard routes.
    source:
        Webcam index (int), video file path, or stream URL (rtsp/http).
    name:
        Human-readable display name.
    width / height / fps:
        Requested capture quality. 0 keeps the source default.
    """

    id: str
    source: int | str
    name: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0


def _get_cameras() -> tuple[CameraConfig, ...]:
    """Parse the camera list from the CAMERAS env var (JSON array).

    Example::

        CAMERAS=[{"id": "cam-north", "name": "North Gate", "source": "rtsp://10.0.0.11/stream"},
                 {"id": "cam-south", "name": "South Gate", "source": "rtsp://10.0.0.12/stream"}]

    Falls back to a single camera from CAMERA_SOURCE (default webcam 0)
    when CAMERAS is not set.
    """
    raw = os.getenv("CAMERAS", "").strip()
    if raw:
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"CAMERAS is not valid JSON: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise ValueError("CAMERAS must be a non-empty JSON array")

        cameras: list[CameraConfig] = []
        for i, entry in enumerate(entries):
            source = entry["source"]
            if isinstance(source, str):
                source = _parse_source(source)
            cam_id = str(entry.get("id", f"cam{i}"))
            cameras.append(
                CameraConfig(
                    id=cam_id,
                    source=source,
                    name=str(entry.get("name", "") or cam_id),
                    width=int(entry.get("width", 0)),
                    height=int(entry.get("height", 0)),
                    fps=float(entry.get("fps", 0)),
                )
            )
        ids = [c.id for c in cameras]
        if len(ids) != len(set(ids)):
            raise ValueError(f"CAMERAS contains duplicate camera ids: {ids}")
        return tuple(cameras)

    source = _parse_source(os.getenv("CAMERA_SOURCE", "0"))
    return (CameraConfig(id="cam0", source=source, name="Camera 0"),)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    All values are resolved once at construction time from environment
    variables (or their defaults).
    """

    cameras: tuple[CameraConfig, ...] = field(default_factory=_get_cameras)
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
    )
    consecutive_hits: int = field(
        default_factory=lambda: int(os.getenv("CONSECUTIVE_HITS", "3"))
    )
    cooldown_seconds: int = field(
        default_factory=lambda: int(os.getenv("COOLDOWN_SECONDS", "5"))
    )
    clip_length: int = field(
        default_factory=lambda: int(os.getenv("CLIP_LENGTH", "90"))
    )
    pre_event_seconds: float = field(
        default_factory=lambda: float(os.getenv("PRE_EVENT_SECONDS", "3"))
    )
    post_event_seconds: float = field(
        default_factory=lambda: float(os.getenv("POST_EVENT_SECONDS", "2"))
    )
    dashboard_port: int = field(
        default_factory=lambda: int(os.getenv("DASHBOARD_PORT", "8000"))
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "data/watchdog.db")
    )
    clip_dir: str = field(
        default_factory=lambda: os.getenv("CLIP_DIR", "data/clips")
    )
    log_dir: str = field(
        default_factory=lambda: os.getenv("LOG_DIR", "logs")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    camera_health_max_age: float = field(
        default_factory=lambda: float(os.getenv("CAMERA_HEALTH_MAX_AGE", "5"))
    )
    object_detection_enabled: bool = field(
        default_factory=lambda: os.getenv("OBJECT_DETECTION", "true").lower()
        in ("1", "true", "yes")
    )
    yolo_model: str = field(
        default_factory=lambda: os.getenv("YOLO_MODEL", "yolov8n.pt")
    )
    yolo_confidence: float = field(
        default_factory=lambda: float(os.getenv("YOLO_CONFIDENCE", "0.4"))
    )
    behavior_enabled: bool = field(
        default_factory=lambda: os.getenv("BEHAVIOR_DETECTION", "true").lower()
        in ("1", "true", "yes")
    )
    loiter_seconds: float = field(
        default_factory=lambda: float(os.getenv("LOITER_SECONDS", "60"))
    )
    run_speed_threshold: float = field(
        default_factory=lambda: float(os.getenv("RUN_SPEED_THRESHOLD", "0.35"))
    )
    anomaly_min_samples: int = field(
        default_factory=lambda: int(os.getenv("ANOMALY_MIN_SAMPLES", "200"))
    )
    behavior_event_cooldown: float = field(
        default_factory=lambda: float(os.getenv("BEHAVIOR_EVENT_COOLDOWN", "30"))
    )


def get_settings() -> Settings:
    """Create a Settings instance after loading the .env file."""
    _load_env()
    return Settings()
