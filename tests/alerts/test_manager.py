"""Tests for src.alerts.manager module."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.alerts.manager import AlertManager
from src.config import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Create Settings with temporary paths for testing."""
    return Settings(
        db_path=str(tmp_path / "test.db"),
        snapshot_dir=str(tmp_path / "snapshots"),
        confidence_threshold=0.85,
        cooldown_seconds=5,
    )


@pytest.fixture()
def manager(settings: Settings) -> AlertManager:
    """Create an AlertManager instance for testing."""
    mgr = AlertManager(settings)
    yield mgr
    mgr.storage.close()


@pytest.fixture()
def fake_frame() -> np.ndarray:
    """Create a fake BGR image frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestAlertManager:
    """Tests for AlertManager."""

    def test_violence_above_threshold_creates_alert(
        self, manager: AlertManager, fake_frame: np.ndarray
    ):
        manager.on_detection("violence", 0.92, fake_frame, "cam0")
        assert manager.alert_count == 1

    def test_violence_below_threshold_ignored(
        self, manager: AlertManager, fake_frame: np.ndarray
    ):
        manager.on_detection("violence", 0.50, fake_frame, "cam0")
        assert manager.alert_count == 0

    def test_normal_label_ignored(
        self, manager: AlertManager, fake_frame: np.ndarray
    ):
        manager.on_detection("normal", 0.95, fake_frame, "cam0")
        assert manager.alert_count == 0

    def test_cooldown_prevents_duplicate_alerts(
        self, settings: Settings, fake_frame: np.ndarray
    ):
        # Use a short cooldown for testing
        short_settings = Settings(
            db_path=settings.db_path,
            snapshot_dir=settings.snapshot_dir,
            confidence_threshold=0.85,
            cooldown_seconds=60,
        )
        mgr = AlertManager(short_settings)

        mgr.on_detection("violence", 0.92, fake_frame, "cam0")
        mgr.on_detection("violence", 0.95, fake_frame, "cam0")

        assert mgr.alert_count == 1
        mgr.storage.close()

    def test_cooldown_expires_allows_new_alert(
        self, fake_frame: np.ndarray, tmp_path: Path
    ):
        short_settings = Settings(
            db_path=str(tmp_path / "test2.db"),
            snapshot_dir=str(tmp_path / "snapshots2"),
            confidence_threshold=0.85,
            cooldown_seconds=0,
        )
        mgr = AlertManager(short_settings)

        mgr.on_detection("violence", 0.92, fake_frame, "cam0")
        mgr.on_detection("violence", 0.95, fake_frame, "cam0")

        assert mgr.alert_count == 2
        mgr.storage.close()

    def test_snapshot_saved_as_jpeg(
        self, manager: AlertManager, fake_frame: np.ndarray, settings: Settings
    ):
        manager.on_detection("violence", 0.92, fake_frame, "cam0")

        snapshot_dir = Path(settings.snapshot_dir)
        jpg_files = list(snapshot_dir.rglob("*.jpg"))
        assert len(jpg_files) == 1
        assert jpg_files[0].suffix == ".jpg"
        assert "cam0" in jpg_files[0].name

    def test_alert_saved_to_database(
        self, manager: AlertManager, fake_frame: np.ndarray
    ):
        manager.on_detection("violence", 0.92, fake_frame, "cam0")

        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["confidence"] == 0.92
        assert alerts[0]["camera_id"] == "cam0"
        assert alerts[0]["status"] == "new"
        assert alerts[0]["snapshot_path"].endswith(".jpg")

    def test_last_alert_time_initially_none(self, manager: AlertManager):
        assert manager.last_alert_time is None

    def test_last_alert_time_updated_after_detection(
        self, manager: AlertManager, fake_frame: np.ndarray
    ):
        manager.on_detection("violence", 0.92, fake_frame, "cam0")
        assert manager.last_alert_time is not None
