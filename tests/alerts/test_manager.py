"""Tests for src.alerts.manager module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.alerts.manager import AlertManager
from src.config import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Create Settings with temporary paths for testing."""
    return Settings(
        db_path=str(tmp_path / "test.db"),
        clip_dir=str(tmp_path / "clips"),
        confidence_threshold=0.85,
        cooldown_seconds=60,
    )


@pytest.fixture()
def manager(settings: Settings) -> AlertManager:
    """Create an AlertManager instance for testing."""
    mgr = AlertManager(settings)
    yield mgr
    mgr.storage.close()


class TestOnClipSaved:
    """Alerts should be created when a clip has been saved."""

    def test_creates_alert(self, manager: AlertManager):
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4", camera_id="cam0")
        assert manager.alert_count == 1

    def test_alert_fields_persisted(self, manager: AlertManager):
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4", camera_id="cam0")

        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["confidence"] == 0.92
        assert alerts[0]["clip_path"] == "clips/a.mp4"
        assert alerts[0]["camera_id"] == "cam0"
        assert alerts[0]["status"] == "new"

    def test_alert_type_persisted(self, manager: AlertManager):
        manager.on_clip_saved(
            confidence=0.7, clip_path="clips/l.mp4",
            camera_id="cam0", alert_type="abnormal_behavior",
        )
        assert manager.get_alerts()[0]["alert_type"] == "abnormal_behavior"

    def test_cooldown_prevents_duplicate_alerts(self, manager: AlertManager):
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4")
        manager.on_clip_saved(confidence=0.95, clip_path="clips/b.mp4")
        assert manager.alert_count == 1

    def test_cooldown_is_per_camera(self, manager: AlertManager):
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4", camera_id="cam0")
        manager.on_clip_saved(confidence=0.95, clip_path="clips/b.mp4", camera_id="cam1")
        assert manager.alert_count == 2

    def test_zero_cooldown_allows_consecutive_alerts(self, tmp_path: Path):
        settings = Settings(
            db_path=str(tmp_path / "test2.db"),
            cooldown_seconds=0,
        )
        mgr = AlertManager(settings)
        mgr.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4")
        mgr.on_clip_saved(confidence=0.95, clip_path="clips/b.mp4")
        assert mgr.alert_count == 2
        mgr.storage.close()

    def test_last_alert_time_initially_none(self, manager: AlertManager):
        assert manager.last_alert_time is None

    def test_last_alert_time_updated_after_alert(self, manager: AlertManager):
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4")
        assert manager.last_alert_time is not None


class TestDeleteAlert:
    """delete_alert should remove the DB row and the clip file."""

    def test_delete_removes_alert_and_clip(self, manager: AlertManager, tmp_path: Path):
        clip_file = tmp_path / "clip.mp4"
        clip_file.write_bytes(b"fake video data")

        manager.on_clip_saved(confidence=0.92, clip_path=str(clip_file))
        alert_id = manager.get_alerts()[0]["id"]

        assert manager.delete_alert(alert_id) is True
        assert manager.alert_count == 0
        assert not clip_file.exists()

    def test_delete_with_missing_clip_still_removes_alert(self, manager: AlertManager):
        manager.on_clip_saved(confidence=0.92, clip_path="does/not/exist.mp4")
        alert_id = manager.get_alerts()[0]["id"]

        assert manager.delete_alert(alert_id) is True
        assert manager.alert_count == 0

    def test_delete_nonexistent_returns_false(self, manager: AlertManager):
        assert manager.delete_alert(9999) is False


class TestGetAlerts:
    """get_alerts should delegate to storage with pagination and filters."""

    def test_get_alerts_pagination(self, tmp_path: Path):
        settings = Settings(
            db_path=str(tmp_path / "test3.db"),
            cooldown_seconds=0,
        )
        mgr = AlertManager(settings)
        for i in range(5):
            mgr.on_clip_saved(confidence=0.9, clip_path=f"clips/{i}.mp4")

        assert len(mgr.get_alerts(limit=2)) == 2
        assert len(mgr.get_alerts(limit=10, offset=4)) == 1
        mgr.storage.close()

    def test_get_alerts_camera_filter(self, tmp_path: Path):
        settings = Settings(
            db_path=str(tmp_path / "test4.db"),
            cooldown_seconds=0,
        )
        mgr = AlertManager(settings)
        mgr.on_clip_saved(confidence=0.9, clip_path="a.mp4", camera_id="cam0")
        mgr.on_clip_saved(confidence=0.9, clip_path="b.mp4", camera_id="cam1")

        cam1_alerts = mgr.get_alerts(camera_id="cam1")
        assert len(cam1_alerts) == 1
        assert cam1_alerts[0]["camera_id"] == "cam1"
        assert mgr.get_alert_count(camera_id="cam0") == 1
        mgr.storage.close()


class TestNotifiers:
    """Registered notifiers should receive every created alert."""

    def test_notifier_called_with_alert(self, manager: AlertManager):
        from unittest.mock import MagicMock

        notifier = MagicMock()
        manager.add_notifier(notifier)
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4", camera_id="cam0")

        notifier.notify.assert_called_once()
        alert, clip_path = notifier.notify.call_args.args
        assert alert["camera_id"] == "cam0"
        assert alert["alert_type"] == "violence"
        assert alert["confidence"] == 0.92
        assert clip_path == "clips/a.mp4"

    def test_failing_notifier_does_not_break_alerting(self, manager: AlertManager):
        from unittest.mock import MagicMock

        bad = MagicMock()
        bad.notify.side_effect = RuntimeError("boom")
        good = MagicMock()
        manager.add_notifier(bad)
        manager.add_notifier(good)

        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4")

        assert manager.alert_count == 1
        good.notify.assert_called_once()

    def test_notifier_not_called_during_cooldown(self, manager: AlertManager):
        from unittest.mock import MagicMock

        notifier = MagicMock()
        manager.add_notifier(notifier)
        manager.on_clip_saved(confidence=0.92, clip_path="clips/a.mp4")
        manager.on_clip_saved(confidence=0.95, clip_path="clips/b.mp4")

        assert notifier.notify.call_count == 1
