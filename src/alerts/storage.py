"""SQLite storage backend for violence detection alerts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class AlertStorage:
    """Manages alert persistence in a SQLite database."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create the alerts table if it does not exist."""
        self._conn.execute(
            """\
            CREATE TABLE IF NOT EXISTS alerts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                snapshot_path TEXT    NOT NULL,
                camera_id     TEXT    NOT NULL,
                status        TEXT    NOT NULL DEFAULT 'new'
            )
            """
        )
        self._conn.commit()

    def save_alert(
        self,
        timestamp: str,
        confidence: float,
        snapshot_path: str,
        camera_id: str,
    ) -> int:
        """Insert a new alert and return its row id."""
        cursor = self._conn.execute(
            "INSERT INTO alerts (timestamp, confidence, snapshot_path, camera_id) "
            "VALUES (?, ?, ?, ?)",
            (timestamp, confidence, snapshot_path, camera_id),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict]:
        """Return alerts ordered by id, with optional status filter."""
        if status is not None:
            cursor = self._conn.execute(
                "SELECT * FROM alerts WHERE status = ? ORDER BY id LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM alerts ORDER BY id LIMIT ? OFFSET ?",
                (limit, offset),
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_alert_count(self, status: str | None = None) -> int:
        """Return total number of alerts, optionally filtered by status."""
        if status is not None:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE status = ?", (status,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM alerts")
        return cursor.fetchone()[0]

    def update_status(self, alert_id: int, status: str) -> bool:
        """Update the status of an alert. Returns True if the alert existed."""
        cursor = self._conn.execute(
            "UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
