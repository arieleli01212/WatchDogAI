"""Application configuration loaded from environment variables."""

from __future__ import annotations

import json
import os
import re
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


def _validate_source(entry: dict, index: int) -> int | str:
    """Extract and validate the 'source' of one CAMERAS entry."""
    if "source" not in entry or entry["source"] is None:
        raise ValueError(f"CAMERAS entry {index}: missing required 'source'")
    source = entry["source"]
    if isinstance(source, str):
        return _parse_source(source)
    if isinstance(source, bool) or not isinstance(source, (int, float)):
        raise ValueError(
            f"CAMERAS entry {index}: source must be a webcam index or URL/path, "
            f"got {source!r}"
        )
    if isinstance(source, float):
        if not source.is_integer():
            raise ValueError(
                f"CAMERAS entry {index}: webcam index must be an integer, got {source}"
            )
        return int(source)
    return source


VIDEO_FILE_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv",
)


def _expand_camera_configs(
    cam_id: str,
    source: int | str,
    name: str,
    width: int,
    height: int,
    fps: float,
) -> list[CameraConfig]:
    """Expand one logical camera into one-or-more CameraConfig entries.

    A plain webcam index, file path, or stream URL produces exactly one
    entry, unchanged. A source that is a *directory* is expanded into one
    entry per video file found directly inside it (non-recursive, sorted
    by filename), each running as its own independent camera pipeline
    with its own id (``<cam_id>-0``, ``<cam_id>-1``, ...), dashboard
    card, and alert attribution — this is how a folder of recorded
    footage is processed alongside live camera streams.
    """
    if not isinstance(source, str) or not os.path.isdir(source):
        return [CameraConfig(id=cam_id, source=source, name=name or cam_id,
                              width=width, height=height, fps=fps)]

    files = sorted(
        f for f in os.listdir(source)
        if os.path.splitext(f)[1].lower() in VIDEO_FILE_EXTENSIONS
    )
    if not files:
        raise ValueError(
            f"Camera {cam_id!r}: source folder {source!r} contains no "
            f"video files ({', '.join(VIDEO_FILE_EXTENSIONS)})"
        )

    base_name = name or cam_id
    return [
        CameraConfig(
            id=f"{cam_id}-{i}",
            source=os.path.join(source, filename),
            name=f"{base_name} ({filename})",
            width=width, height=height, fps=fps,
        )
        for i, filename in enumerate(files)
    ]


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
                 {"id": "cam-south", "name": "South Gate", "source": "rtsp://10.0.0.12/stream"},
                 {"id": "archive", "name": "Archived footage", "source": "C:/footage/2026-07-05"}]

    A ``source`` that is a directory path expands into one camera per
    video file found inside it (ids ``archive-0``, ``archive-1``, ...),
    letting live camera streams and a folder of recorded video files run
    side by side in the same deployment.

    Falls back to a single camera from CAMERA_SOURCE (default webcam 0)
    when CAMERAS is not set; CAMERA_SOURCE also accepts a folder path.
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
            cam_id = str(entry.get("id", f"cam{i}"))
            # Camera ids end up in filenames, URLs, and MQTT topics
            if not re.fullmatch(r"[A-Za-z0-9_-]+", cam_id):
                raise ValueError(
                    f"CAMERAS entry {i}: id {cam_id!r} may only contain "
                    "letters, digits, '-' and '_'"
                )
            cameras.extend(
                _expand_camera_configs(
                    cam_id=cam_id,
                    source=_validate_source(entry, i),
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
    return tuple(_expand_camera_configs(
        cam_id="cam0", source=source, name="Camera 0", width=0, height=0, fps=0.0,
    ))


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
    pre_event_seconds: float = field(
        default_factory=lambda: float(os.getenv("PRE_EVENT_SECONDS", "3"))
    )
    post_event_seconds: float = field(
        default_factory=lambda: float(os.getenv("POST_EVENT_SECONDS", "2"))
    )
    max_clip_seconds: float = field(
        default_factory=lambda: float(os.getenv("MAX_CLIP_SECONDS", "30"))
    )
    dashboard_port: int = field(
        default_factory=lambda: int(os.getenv("DASHBOARD_PORT", "8000"))
    )
    db_backend: str = field(
        default_factory=lambda: os.getenv("DB_BACKEND", "auto")
    )
    mongodb_uri: str = field(
        default_factory=lambda: os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    )
    mongodb_db: str = field(
        default_factory=lambda: os.getenv("MONGODB_DB", "watchdog")
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
    record_behavior_clips: bool = field(
        default_factory=lambda: os.getenv("RECORD_BEHAVIOR_CLIPS", "false").lower()
        in ("1", "true", "yes")
    )
    control_center_url: str = field(
        default_factory=lambda: os.getenv("CONTROL_CENTER_URL", "")
    )
    control_center_api_key: str = field(
        default_factory=lambda: os.getenv("CONTROL_CENTER_API_KEY", "")
    )
    mqtt_host: str = field(
        default_factory=lambda: os.getenv("MQTT_HOST", "")
    )
    mqtt_port: int = field(
        default_factory=lambda: int(os.getenv("MQTT_PORT", "1883"))
    )
    mqtt_username: str = field(
        default_factory=lambda: os.getenv("MQTT_USERNAME", "")
    )
    mqtt_password: str = field(
        default_factory=lambda: os.getenv("MQTT_PASSWORD", "")
    )
    mqtt_base_topic: str = field(
        default_factory=lambda: os.getenv("MQTT_BASE_TOPIC", "watchdog")
    )
    telemetry_interval: float = field(
        default_factory=lambda: float(os.getenv("TELEMETRY_INTERVAL", "30"))
    )
    api_token: str = field(
        default_factory=lambda: os.getenv("API_TOKEN", "")
    )


def get_settings() -> Settings:
    """Create a Settings instance after loading the .env file."""
    _load_env()
    return Settings()
