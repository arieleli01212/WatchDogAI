"""Alert manager: coordinates detection events, snapshots, and storage."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.alerts.storage import AlertStorage
from src.config import Settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Orchestrates alert creation from detection events.

    Handles cooldown enforcement, snapshot saving, and database persistence.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.storage = AlertStorage(settings.db_path)
        self._snapshot_dir = Path(settings.snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_alert_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_detection(
        self,
        label: str,
        confidence: float,
        frame: np.ndarray,
        camera_id: str = "cam0",
    ) -> None:
        """Process a detection result. Creates an alert when appropriate."""
        if label != "violence":
            return
        if confidence < self._settings.confidence_threshold:
            return
        if self._is_in_cooldown():
            return

        now = datetime.now(tz=timezone.utc)
        snapshot_path = self._save_snapshot(frame, now, camera_id)

        self.storage.save_alert(
            timestamp=now.isoformat(),
            confidence=confidence,
            snapshot_path=str(snapshot_path),
            camera_id=camera_id,
        )

        self._last_alert_time = time.monotonic()
        logger.info(
            "Alert created: camera=%s confidence=%.2f snapshot=%s",
            camera_id,
            confidence,
            snapshot_path,
        )

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict]:
        """Delegate to storage."""
        return self.storage.get_alerts(limit=limit, offset=offset, status=status)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alert_count(self) -> int:
        """Total number of alerts in the database."""
        return self.storage.get_alert_count()

    @property
    def last_alert_time(self) -> float | None:
        """Monotonic timestamp of the most recent alert, or None."""
        return self._last_alert_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_in_cooldown(self) -> bool:
        if self._last_alert_time is None:
            return False
        elapsed = time.monotonic() - self._last_alert_time
        return elapsed < self._settings.cooldown_seconds

    def _save_snapshot(
        self,
        frame: np.ndarray,
        now: datetime,
        camera_id: str,
    ) -> Path:
        date_dir = self._snapshot_dir / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{now.strftime('%H-%M-%S')}_{camera_id}.jpg"
        path = date_dir / filename
        cv2.imwrite(str(path), frame)
        return path
