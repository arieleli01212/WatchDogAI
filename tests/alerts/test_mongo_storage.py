"""Tests for the MongoDB alert storage backend (using mongomock)."""

from __future__ import annotations

import mongomock
import pytest

from src.alerts.mongo_storage import MongoAlertStorage


@pytest.fixture()
def storage() -> MongoAlertStorage:
    """MongoAlertStorage backed by an in-memory mongomock client."""
    return MongoAlertStorage(client=mongomock.MongoClient())


class TestMongoAlertStorage:
    """Behavioral parity with the SQLite backend."""

    def test_save_alert_returns_incrementing_int_ids(self, storage):
        id1 = storage.save_alert("2026-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        id2 = storage.save_alert("2026-01-15T10:31:00", 0.88, "clip2.mp4", "cam0")
        assert isinstance(id1, int)
        assert id1 >= 1
        assert id2 > id1

    def test_get_alerts_newest_first(self, storage):
        storage.save_alert("2026-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        storage.save_alert("2026-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")

        alerts = storage.get_alerts()
        assert len(alerts) == 2
        assert alerts[0]["camera_id"] == "cam1"
        assert alerts[1]["camera_id"] == "cam0"
        # Internal Mongo _id must not leak into the API payload
        assert "_id" not in alerts[0]

    def test_get_alerts_limit_offset(self, storage):
        for i in range(10):
            storage.save_alert(f"2026-01-15T10:3{i}:00", 0.9, f"c{i}.mp4", "cam0")

        assert len(storage.get_alerts(limit=3)) == 3
        assert len(storage.get_alerts(limit=3, offset=8)) == 2

    def test_status_and_camera_filters(self, storage):
        storage.save_alert("2026-01-15T10:30:00", 0.92, "clip1.mp4", "cam0")
        id2 = storage.save_alert("2026-01-15T10:31:00", 0.88, "clip2.mp4", "cam1")
        storage.update_status(id2, "acknowledged")

        assert len(storage.get_alerts(status="new")) == 1
        assert len(storage.get_alerts(camera_id="cam1")) == 1
        assert storage.get_alert_count() == 2
        assert storage.get_alert_count(status="acknowledged") == 1
        assert storage.get_alert_count(camera_id="cam0") == 1

    def test_alert_type_persisted(self, storage):
        storage.save_alert(
            "2026-01-15T10:30:00", 0.7, "clip1.mp4", "cam0",
            alert_type="loitering",
        )
        assert storage.get_alerts()[0]["alert_type"] == "loitering"

    def test_get_alert_by_id(self, storage):
        alert_id = storage.save_alert("2026-01-15T10:30:00", 0.92, "c.mp4", "cam0")
        alert = storage.get_alert(alert_id)
        assert alert is not None
        assert alert["clip_path"] == "c.mp4"
        assert storage.get_alert(9999) is None

    def test_update_status(self, storage):
        alert_id = storage.save_alert("2026-01-15T10:30:00", 0.92, "c.mp4", "cam0")
        assert storage.update_status(alert_id, "reviewed") is True
        assert storage.get_alert(alert_id)["status"] == "reviewed"
        assert storage.update_status(9999, "reviewed") is False

    def test_delete_alert(self, storage):
        alert_id = storage.save_alert("2026-01-15T10:30:00", 0.92, "c.mp4", "cam0")
        assert storage.delete_alert(alert_id) is True
        assert storage.get_alert(alert_id) is None
        assert storage.delete_alert(alert_id) is False


class TestStorageFactory:
    """create_alert_storage should honor the configured backend."""

    def test_sqlite_backend(self, tmp_path):
        from src.alerts.storage import AlertStorage, create_alert_storage
        from src.config import Settings

        settings = Settings(
            db_backend="sqlite", db_path=str(tmp_path / "a.db")
        )
        storage = create_alert_storage(settings)
        assert isinstance(storage, AlertStorage)
        storage.close()

    def test_auto_falls_back_to_sqlite_when_mongo_unreachable(self, tmp_path):
        from unittest.mock import patch

        from src.alerts.storage import AlertStorage, create_alert_storage
        from src.config import Settings

        settings = Settings(db_backend="auto", db_path=str(tmp_path / "a.db"))
        with patch(
            "src.alerts.mongo_storage.MongoAlertStorage",
            side_effect=ConnectionError("no mongo"),
        ):
            storage = create_alert_storage(settings)
        assert isinstance(storage, AlertStorage)
        storage.close()

    def test_explicit_mongodb_backend_raises_when_unreachable(self, tmp_path):
        from unittest.mock import patch

        from src.alerts.storage import create_alert_storage
        from src.config import Settings

        settings = Settings(db_backend="mongodb", db_path=str(tmp_path / "a.db"))
        with patch(
            "src.alerts.mongo_storage.MongoAlertStorage",
            side_effect=ConnectionError("no mongo"),
        ):
            with pytest.raises(ConnectionError):
                create_alert_storage(settings)

    def test_auto_uses_mongo_when_available(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from src.alerts.storage import create_alert_storage
        from src.config import Settings

        settings = Settings(db_backend="auto", db_path=str(tmp_path / "a.db"))
        fake_storage = MagicMock()
        with patch(
            "src.alerts.mongo_storage.MongoAlertStorage",
            return_value=fake_storage,
        ) as mock_cls:
            storage = create_alert_storage(settings)
        assert storage is fake_storage
        mock_cls.assert_called_once_with(
            uri=settings.mongodb_uri, db_name=settings.mongodb_db
        )
