"""Tests for src.alerts.storage module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.alerts.storage import AlertStorage


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    """Return a temporary database path."""
    return str(tmp_path / "test_alerts.db")


@pytest.fixture()
def storage(db_path: str) -> AlertStorage:
    """Create an AlertStorage instance with a temporary database."""
    s = AlertStorage(db_path)
    yield s
    s.close()


class TestAlertStorage:
    """Tests for AlertStorage SQLite operations."""

    def test_creates_database_and_table(self, db_path: str):
        storage = AlertStorage(db_path)
        assert Path(db_path).exists()

        # Verify table schema
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(alerts)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()
        storage.close()

        assert "id" in columns
        assert "timestamp" in columns
        assert "confidence" in columns
        assert "clip_path" in columns
        assert "camera_id" in columns
        assert "alert_type" in columns
        assert "status" in columns

    def test_migrates_old_schema_without_alert_type(self, tmp_path: Path):
        """Databases created before alert types existed gain the column."""
        old_db = str(tmp_path / "old.db")
        conn = sqlite3.connect(old_db)
        conn.execute(
            """CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                clip_path TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new'
            )"""
        )
        conn.execute(
            "INSERT INTO alerts (timestamp, confidence, clip_path, camera_id) "
            "VALUES ('2025-01-15T10:30:00', 0.9, 'clip.mp4', 'cam0')"
        )
        conn.commit()
        conn.close()

        storage = AlertStorage(old_db)
        alerts = storage.get_alerts()
        assert alerts[0]["alert_type"] == "violence"
        storage.close()

    def test_alert_type_defaults_to_violence(self, storage: AlertStorage):
        storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        assert storage.get_alerts()[0]["alert_type"] == "violence"

    def test_alert_type_persisted(self, storage: AlertStorage):
        storage.save_alert(
            "2025-01-15T10:30:00", 0.7, "clip1.mp4", "cam0",
            alert_type="abnormal_behavior",
        )
        assert storage.get_alerts()[0]["alert_type"] == "abnormal_behavior"

    def test_get_alerts_camera_filter(self, storage: AlertStorage):
        storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        storage.save_alert("2025-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")

        cam0 = storage.get_alerts(camera_id="cam0")
        assert len(cam0) == 1
        assert cam0[0]["camera_id"] == "cam0"
        assert storage.get_alert_count(camera_id="cam1") == 1

    def test_save_alert_returns_id(self, storage: AlertStorage):
        alert_id = storage.save_alert(
            timestamp="2025-01-15T10:30:00",
            confidence=0.92,
            clip_path="data/clips/2025-01-15/10-30-00_cam0.mp4",
            camera_id="cam0",
        )
        assert isinstance(alert_id, int)
        assert alert_id >= 1

    def test_save_alert_returns_incrementing_ids(self, storage: AlertStorage):
        id1 = storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        id2 = storage.save_alert("2025-01-15T10:31:00", 0.88, "clip2.mp4", "cam0")
        assert id2 > id1

    def test_get_alerts_returns_saved_alerts(self, storage: AlertStorage):
        storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        storage.save_alert("2025-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")

        alerts = storage.get_alerts()
        # Newest first (ordered by id desc)
        assert len(alerts) == 2
        assert alerts[0]["confidence"] == 0.88
        assert alerts[0]["camera_id"] == "cam1"
        assert alerts[1]["confidence"] == 0.92
        assert alerts[1]["camera_id"] == "cam0"

    def test_get_alerts_respects_limit_and_offset(self, storage: AlertStorage):
        for i in range(10):
            storage.save_alert(f"2025-01-15T10:3{i}:00", 0.9, f"clip{i}.mp4", "cam0")

        alerts = storage.get_alerts(limit=3, offset=0)
        assert len(alerts) == 3

        alerts = storage.get_alerts(limit=3, offset=8)
        assert len(alerts) == 2

    def test_get_alerts_with_status_filter(self, storage: AlertStorage):
        storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        id2 = storage.save_alert("2025-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")
        storage.update_status(id2, "acknowledged")

        new_alerts = storage.get_alerts(status="new")
        assert len(new_alerts) == 1
        assert new_alerts[0]["status"] == "new"

        ack_alerts = storage.get_alerts(status="acknowledged")
        assert len(ack_alerts) == 1
        assert ack_alerts[0]["status"] == "acknowledged"

    def test_get_alert_returns_single_alert(self, storage: AlertStorage):
        alert_id = storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        alert = storage.get_alert(alert_id)
        assert alert is not None
        assert alert["clip_path"] == "clip1.mp4"

    def test_get_alert_nonexistent_returns_none(self, storage: AlertStorage):
        assert storage.get_alert(9999) is None

    def test_get_alert_count(self, storage: AlertStorage):
        assert storage.get_alert_count() == 0
        storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        storage.save_alert("2025-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")
        assert storage.get_alert_count() == 2

    def test_get_alert_count_with_status_filter(self, storage: AlertStorage):
        storage.save_alert("2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        id2 = storage.save_alert("2025-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")
        storage.update_status(id2, "acknowledged")

        assert storage.get_alert_count(status="new") == 1
        assert storage.get_alert_count(status="acknowledged") == 1

    def test_update_status(self, storage: AlertStorage):
        alert_id = storage.save_alert(
            "2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0"
        )
        result = storage.update_status(alert_id, "acknowledged")
        assert result is True

        alerts = storage.get_alerts()
        assert alerts[0]["status"] == "acknowledged"

    def test_update_status_nonexistent_returns_false(self, storage: AlertStorage):
        result = storage.update_status(9999, "acknowledged")
        assert result is False

    def test_delete_alert(self, storage: AlertStorage):
        alert_id = storage.save_alert(
            "2025-01-15T10:30:00", 0.92, "clip1.mp4", "cam0"
        )
        assert storage.delete_alert(alert_id) is True
        assert storage.get_alert(alert_id) is None

    def test_delete_alert_nonexistent_returns_false(self, storage: AlertStorage):
        assert storage.delete_alert(9999) is False
