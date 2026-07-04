"""MongoDB storage backend for detection alerts."""

from __future__ import annotations

import logging

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument

logger = logging.getLogger(__name__)


class MongoAlertStorage:
    """Persists alerts in a MongoDB collection.

    Mirrors the AlertStorage (SQLite) interface so the two backends are
    interchangeable. Alerts keep integer ``id`` fields (assigned from an
    atomic counter document) so the REST API and dashboard behave
    identically on both backends.
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "watchdog",
        collection: str = "alerts",
        client: MongoClient | None = None,
        timeout_ms: int = 1500,
    ) -> None:
        if client is None:
            client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            # Fail fast when the server is unreachable so callers can fall back
            client.admin.command("ping")
        self._client = client
        self._db = client[db_name]
        self._alerts = self._db[collection]
        self._counters = self._db["counters"]

        self._alerts.create_index([("id", ASCENDING)], unique=True)
        self._alerts.create_index([("camera_id", ASCENDING)])
        self._alerts.create_index([("status", ASCENDING)])

    def _next_id(self) -> int:
        doc = self._counters.find_one_and_update(
            {"_id": "alerts"},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return int(doc["seq"])

    def save_alert(
        self,
        timestamp: str,
        confidence: float,
        clip_path: str,
        camera_id: str,
        alert_type: str = "violence",
    ) -> int:
        """Insert a new alert and return its integer id."""
        alert_id = self._next_id()
        self._alerts.insert_one(
            {
                "id": alert_id,
                "timestamp": timestamp,
                "confidence": confidence,
                "clip_path": clip_path,
                "camera_id": camera_id,
                "alert_type": alert_type,
                "status": "new",
            }
        )
        return alert_id

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> list[dict]:
        """Return alerts ordered newest-first, with optional filters."""
        query: dict = {}
        if status is not None:
            query["status"] = status
        if camera_id is not None:
            query["camera_id"] = camera_id
        cursor = (
            self._alerts.find(query, {"_id": 0})
            .sort("id", DESCENDING)
            .skip(offset)
            .limit(limit)
        )
        return list(cursor)

    def get_alert(self, alert_id: int) -> dict | None:
        """Return a single alert by id, or None if not found."""
        return self._alerts.find_one({"id": alert_id}, {"_id": 0})

    def get_alert_count(
        self,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> int:
        """Return total number of alerts, optionally filtered."""
        query: dict = {}
        if status is not None:
            query["status"] = status
        if camera_id is not None:
            query["camera_id"] = camera_id
        return self._alerts.count_documents(query)

    def update_status(self, alert_id: int, status: str) -> bool:
        """Update the status of an alert. Returns True if the alert existed."""
        result = self._alerts.update_one(
            {"id": alert_id}, {"$set": {"status": status}}
        )
        return result.matched_count > 0

    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert by id. Returns True if the alert existed."""
        result = self._alerts.delete_one({"id": alert_id})
        return result.deleted_count > 0

    def close(self) -> None:
        """Close the client connection."""
        self._client.close()
