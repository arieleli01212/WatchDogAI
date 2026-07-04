"""Tests for src.config module."""

from __future__ import annotations

import pytest

from src.config import CameraConfig, Settings, get_settings


class TestSettingsDefaults:
    """Settings should have sensible defaults when no env vars are set."""

    def test_default_single_camera(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.delenv("CAMERA_SOURCE", raising=False)
        settings = Settings()
        assert len(settings.cameras) == 1
        assert settings.cameras[0].id == "cam0"
        assert settings.cameras[0].source == 0

    def test_default_confidence_threshold(self):
        settings = Settings()
        assert settings.confidence_threshold == 0.85

    def test_default_consecutive_hits(self):
        settings = Settings()
        assert settings.consecutive_hits == 3

    def test_default_cooldown_seconds(self):
        settings = Settings()
        assert settings.cooldown_seconds == 5

    def test_default_pre_event_seconds(self):
        settings = Settings()
        assert settings.pre_event_seconds == 3

    def test_default_post_event_seconds(self):
        settings = Settings()
        assert settings.post_event_seconds == 2

    def test_default_dashboard_port(self):
        settings = Settings()
        assert settings.dashboard_port == 8000

    def test_default_db_path(self):
        settings = Settings()
        assert settings.db_path == "data/watchdog.db"

    def test_default_clip_dir(self):
        settings = Settings()
        assert settings.clip_dir == "data/clips"

    def test_default_log_dir(self):
        settings = Settings()
        assert settings.log_dir == "logs"

    def test_default_log_level(self):
        settings = Settings()
        assert settings.log_level == "INFO"


class TestSettingsFromEnv:
    """Settings should respect environment variable overrides."""

    def test_confidence_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.5")
        settings = Settings()
        assert settings.confidence_threshold == 0.5

    def test_consecutive_hits_from_env(self, monkeypatch):
        monkeypatch.setenv("CONSECUTIVE_HITS", "5")
        settings = Settings()
        assert settings.consecutive_hits == 5

    def test_cooldown_seconds_from_env(self, monkeypatch):
        monkeypatch.setenv("COOLDOWN_SECONDS", "10")
        settings = Settings()
        assert settings.cooldown_seconds == 10

    def test_dashboard_port_from_env(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PORT", "9000")
        settings = Settings()
        assert settings.dashboard_port == 9000

    def test_camera_source_int_from_env(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", "2")
        settings = Settings()
        assert settings.cameras[0].source == 2

    def test_camera_source_path_from_env(self, monkeypatch):
        monkeypatch.delenv("CAMERAS", raising=False)
        monkeypatch.setenv("CAMERA_SOURCE", "/dev/video1")
        settings = Settings()
        assert settings.cameras[0].source == "/dev/video1"

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_db_path_from_env(self, monkeypatch):
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        settings = Settings()
        assert settings.db_path == "/tmp/test.db"

    def test_clip_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("CLIP_DIR", "/tmp/clips")
        settings = Settings()
        assert settings.clip_dir == "/tmp/clips"

    def test_log_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_DIR", "/tmp/logs")
        settings = Settings()
        assert settings.log_dir == "/tmp/logs"

    def test_pre_event_seconds_from_env(self, monkeypatch):
        monkeypatch.setenv("PRE_EVENT_SECONDS", "5")
        settings = Settings()
        assert settings.pre_event_seconds == 5.0


class TestCamerasFromEnv:
    """The CAMERAS env var should configure multiple cameras from JSON."""

    def test_two_cameras_from_json(self, monkeypatch):
        monkeypatch.setenv(
            "CAMERAS",
            '[{"id": "cam-north", "name": "North Gate", "source": "rtsp://10.0.0.11/stream"},'
            ' {"id": "cam-south", "name": "South Gate", "source": 1}]',
        )
        settings = Settings()
        assert len(settings.cameras) == 2
        north, south = settings.cameras
        assert north.id == "cam-north"
        assert north.name == "North Gate"
        assert north.source == "rtsp://10.0.0.11/stream"
        assert south.id == "cam-south"
        assert south.source == 1

    def test_camera_quality_settings(self, monkeypatch):
        monkeypatch.setenv(
            "CAMERAS",
            '[{"id": "c1", "source": 0, "width": 1280, "height": 720, "fps": 15}]',
        )
        settings = Settings()
        cam = settings.cameras[0]
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 15.0

    def test_camera_ids_default_to_index(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"source": 0}, {"source": 1}]')
        settings = Settings()
        assert [c.id for c in settings.cameras] == ["cam0", "cam1"]

    def test_numeric_string_source_becomes_int(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": "0"}]')
        settings = Settings()
        assert settings.cameras[0].source == 0

    def test_invalid_json_raises(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", "not json")
        with pytest.raises(ValueError):
            Settings()

    def test_duplicate_ids_raise(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", '[{"id": "c1", "source": 0}, {"id": "c1", "source": 1}]')
        with pytest.raises(ValueError):
            Settings()

    def test_empty_list_raises(self, monkeypatch):
        monkeypatch.setenv("CAMERAS", "[]")
        with pytest.raises(ValueError):
            Settings()

    def test_explicit_cameras_override(self):
        cameras = (CameraConfig(id="a", source=0), CameraConfig(id="b", source=1))
        settings = Settings(cameras=cameras)
        assert settings.cameras == cameras


class TestGetSettings:
    """get_settings() should return a valid Settings instance."""

    def test_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_settings_is_frozen(self):
        settings = get_settings()
        with pytest.raises(AttributeError):
            settings.log_level = "DEBUG"  # type: ignore[misc]
