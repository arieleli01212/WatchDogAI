"""Push alerts with video clips to the municipal control center over HTTP."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_STOP = object()


class ControlCenterNotifier:
    """Delivers alerts (JSON + MP4 clip) to a control-center endpoint.

    Deliveries run on a background worker with a bounded queue and
    exponential-backoff retries, so a slow or unreachable control center
    never blocks the analysis pipelines and short outages lose nothing.

    The alert is POSTed as multipart/form-data: an ``alert`` field with
    the JSON payload and a ``clip`` field with the MP4 file. If the clip
    file is missing the alert is sent as plain JSON instead.
    """

    def __init__(
        self,
        url: str,
        api_key: str = "",
        timeout: float = 10.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        queue_size: int = 100,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._worker = threading.Thread(
            target=self._run, name="control-center-notifier", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Notifier interface (called by AlertManager)
    # ------------------------------------------------------------------

    def notify(self, alert: dict, clip_path: str) -> None:
        """Queue an alert for delivery. Never blocks the caller."""
        try:
            self._queue.put_nowait((alert, clip_path))
        except queue.Full:
            logger.error(
                "Control-center queue full (%d pending) — dropping alert %s",
                self._queue.maxsize, alert.get("id"),
            )

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker after the queued deliveries finish."""
        self._queue.put(_STOP)
        self._worker.join(timeout)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            alert, clip_path = item
            self._deliver(alert, clip_path)
            self._queue.task_done()

    def _deliver(self, alert: dict, clip_path: str) -> None:
        for attempt in range(1, self._max_retries + 1):
            try:
                self._post(alert, clip_path)
                logger.info(
                    "Alert %s delivered to control center (%s)",
                    alert.get("id"), self._url,
                )
                return
            except Exception as exc:
                delay = self._backoff_base ** (attempt - 1)
                logger.warning(
                    "Control-center delivery failed (attempt %d/%d): %s — "
                    "retrying in %.0fs",
                    attempt, self._max_retries, exc, delay,
                )
                time.sleep(delay)
        logger.error(
            "Giving up on alert %s after %d attempts",
            alert.get("id"), self._max_retries,
        )

    def _post(self, alert: dict, clip_path: str) -> None:
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        clip = Path(clip_path)
        if clip.is_file():
            with open(clip, "rb") as fh:
                response = requests.post(
                    self._url,
                    data={"alert": json.dumps(alert)},
                    files={"clip": (clip.name, fh, "video/mp4")},
                    headers=headers,
                    timeout=self._timeout,
                )
        else:
            logger.warning(
                "Clip %s missing — sending alert %s without video",
                clip_path, alert.get("id"),
            )
            response = requests.post(
                self._url, json=alert, headers=headers, timeout=self._timeout
            )
        response.raise_for_status()
