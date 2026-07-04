"""SQLite storage backend for detection alerts."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class AlertStorage:
    """Manages alert persistence in a SQLite database.

    A single connection is shared across the clip-writer threads and the
    dashboard, so every operation is serialized with a lock.
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the alerts table if it does not exist and migrate old schemas."""
        with self._lock:
            self._conn.execute(
                """\
                CREATE TABLE IF NOT EXISTS alerts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    confidence    REAL    NOT NULL,
                    clip_path     TEXT    NOT NULL,
                    camera_id     TEXT    NOT NULL,
                    alert_type    TEXT    NOT NULL DEFAULT 'violence',
                    status        TEXT    NOT NULL DEFAULT 'new'
                )
                """
            )
            # Databases created before alert types existed lack the column
            columns = {
                row[1] for row in self._conn.execute("PRAGMA table_info(alerts)")
            }
            if "alert_type" not in columns:
                self._conn.execute(
                    "ALTER TABLE alerts ADD COLUMN alert_type TEXT NOT NULL DEFAULT 'violence'"
                )
            self._conn.commit()

    def save_alert(
        self,
        timestamp: str,
        confidence: float,
        clip_path: str,
        camera_id: str,
        alert_type: str = "violence",
    ) -> int:
        """Insert a new alert and return its row id."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO alerts (timestamp, confidence, clip_path, camera_id, alert_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, confidence, clip_path, camera_id, alert_type),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> list[dict]:
        """Return alerts ordered by id desc, with optional filters."""
        query = "SELECT * FROM alerts"
        clauses: list[str] = []
        params: list = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            cursor = self._conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_alert(self, alert_id: int) -> dict | None:
        """Return a single alert by id, or None if not found."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_alert_count(
        self,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> int:
        """Return total number of alerts, optionally filtered."""
        query = "SELECT COUNT(*) FROM alerts"
        clauses: list[str] = []
        params: list = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        with self._lock:
            cursor = self._conn.execute(query, params)
            return cursor.fetchone()[0]

    def update_status(self, alert_id: int, status: str) -> bool:
        """Update the status of an alert. Returns True if the alert existed."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert by id. Returns True if the alert existed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM alerts WHERE id = ?", (alert_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
