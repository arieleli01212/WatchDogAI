"""Alert manager: coordinates clip saving, storage, and notification."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src.alerts.storage import AlertStorage
from src.config import Settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Orchestrates alert creation from clip events.

    Handles per-camera cooldown enforcement, database persistence, and
    alert deletion. Notifiers (control-center push, MQTT) can be attached
    with :meth:`add_notifier` and are invoked for every created alert.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.storage = AlertStorage(settings.db_path)
        self._last_alert_times: dict[str, float] = {}
        self._notifiers: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_notifier(self, notifier) -> None:
        """Register a notifier with a ``notify(alert: dict, clip_path: str)`` method."""
        self._notifiers.append(notifier)

    def on_clip_saved(
        self,
        confidence: float,
        clip_path: str,
        camera_id: str = "cam0",
        alert_type: str = "violence",
    ) -> None:
        """Create an alert after a clip has been written to disk."""
        if self._is_in_cooldown(camera_id):
            logger.debug("Alert suppressed by cooldown: camera=%s", camera_id)
            return

        now = datetime.now(tz=timezone.utc)
        alert_id = self.storage.save_alert(
            timestamp=now.isoformat(),
            confidence=confidence,
            clip_path=clip_path,
            camera_id=camera_id,
            alert_type=alert_type,
        )

        self._last_alert_times[camera_id] = time.monotonic()
        logger.info(
            "Alert created: id=%s type=%s camera=%s confidence=%.2f clip=%s",
            alert_id, alert_type, camera_id, confidence, clip_path,
        )

        alert = {
            "id": alert_id,
            "timestamp": now.isoformat(),
            "confidence": confidence,
            "clip_path": clip_path,
            "camera_id": camera_id,
            "alert_type": alert_type,
            "status": "new",
        }
        for notifier in self._notifiers:
            try:
                notifier.notify(alert, clip_path)
            except Exception:
                logger.exception(
                    "Notifier %s failed for alert %s",
                    type(notifier).__name__, alert_id,
                )

    def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert and its clip file from disk."""
        alert = self.storage.get_alert(alert_id)
        if alert is None:
            return False

        # Remove the clip file if it exists
        clip_file = Path(alert["clip_path"])
        if clip_file.is_file():
            clip_file.unlink()
            logger.info("Deleted clip file: %s", clip_file)

        self.storage.delete_alert(alert_id)
        logger.info("Deleted alert id=%s", alert_id)
        return True

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> list[dict]:
        """Delegate to storage."""
        return self.storage.get_alerts(
            limit=limit, offset=offset, status=status, camera_id=camera_id
        )

    def get_alert_count(
        self,
        status: str | None = None,
        camera_id: str | None = None,
    ) -> int:
        """Delegate to storage."""
        return self.storage.get_alert_count(status=status, camera_id=camera_id)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alert_count(self) -> int:
        """Total number of alerts in the database."""
        return self.storage.get_alert_count()

    @property
    def last_alert_time(self) -> float | None:
        """Monotonic timestamp of the most recent alert across all cameras."""
        if not self._last_alert_times:
            return None
        return max(self._last_alert_times.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, camera_id: str) -> bool:
        last = self._last_alert_times.get(camera_id)
        if last is None:
            return False
        elapsed = time.monotonic() - last
        return elapsed < self._settings.cooldown_seconds
