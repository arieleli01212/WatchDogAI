"""Tests for src.config module."""

from __future__ import annotations

import os

import pytest

from src.config import Settings, get_settings


class TestSettingsDefaults:
    """Settings should have sensible defaults when no env vars are set."""

    def test_default_camera_source(self):
        settings = Settings()
        assert settings.camera_source == 0

    def test_default_confidence_threshold(self):
        settings = Settings()
        assert settings.confidence_threshold == 0.85

    def test_default_cooldown_seconds(self):
        settings = Settings()
        assert settings.cooldown_seconds == 5

    def test_default_clip_length(self):
        settings = Settings()
        assert settings.clip_length == 16

    def test_default_dashboard_port(self):
        settings = Settings()
        assert settings.dashboard_port == 8000

    def test_default_model_path(self):
        settings = Settings()
        assert settings.model_path == "models/violence_detector.pt"

    def test_default_db_path(self):
        settings = Settings()
        assert settings.db_path == "data/watchdog.db"

    def test_default_snapshot_dir(self):
        settings = Settings()
        assert settings.snapshot_dir == "data/snapshots"

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

    def test_cooldown_seconds_from_env(self, monkeypatch):
        monkeypatch.setenv("COOLDOWN_SECONDS", "10")
        settings = Settings()
        assert settings.cooldown_seconds == 10

    def test_dashboard_port_from_env(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_PORT", "9000")
        settings = Settings()
        assert settings.dashboard_port == 9000

    def test_camera_source_int_from_env(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SOURCE", "2")
        settings = Settings()
        assert settings.camera_source == 2

    def test_camera_source_path_from_env(self, monkeypatch):
        monkeypatch.setenv("CAMERA_SOURCE", "/dev/video1")
        settings = Settings()
        assert settings.camera_source == "/dev/video1"

    def test_model_path_from_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", "/custom/model.pt")
        settings = Settings()
        assert settings.model_path == "/custom/model.pt"

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings()
        assert settings.log_level == "DEBUG"

    def test_clip_length_from_env(self, monkeypatch):
        monkeypatch.setenv("CLIP_LENGTH", "32")
        settings = Settings()
        assert settings.clip_length == 32

    def test_db_path_from_env(self, monkeypatch):
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")
        settings = Settings()
        assert settings.db_path == "/tmp/test.db"

    def test_snapshot_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("SNAPSHOT_DIR", "/tmp/snaps")
        settings = Settings()
        assert settings.snapshot_dir == "/tmp/snaps"

    def test_log_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_DIR", "/tmp/logs")
        settings = Settings()
        assert settings.log_dir == "/tmp/logs"


class TestGetSettings:
    """get_settings() should return a valid Settings instance."""

    def test_returns_settings_instance(self):
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_settings_is_frozen(self):
        settings = get_settings()
        with pytest.raises(AttributeError):
            settings.log_level = "DEBUG"  # type: ignore[misc]
