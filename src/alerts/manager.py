"""Alert manager: coordinates clip saving and storage."""

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

    Handles cooldown enforcement, database persistence, and alert deletion.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.storage = AlertStorage(settings.db_path)
        self._last_alert_time: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_clip_saved(
        self,
        confidence: float,
        clip_path: str,
        camera_id: str = "cam0",
    ) -> None:
        """Create an alert after a clip has been written to disk."""
        if self._is_in_cooldown():
            return

        now = datetime.now(tz=timezone.utc)
        self.storage.save_alert(
            timestamp=now.isoformat(),
            confidence=confidence,
            clip_path=clip_path,
            camera_id=camera_id,
        )

        self._last_alert_time = time.monotonic()
        logger.info(
            "Alert created: camera=%s confidence=%.2f clip=%s",
            camera_id,
            confidence,
            clip_path,
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
        logger.info("Deleted alert id=%d", alert_id)
        return True

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
