"""Push alerts with video clips to the municipal control center over HTTP."""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

import requests

# Bound the real exception class at import time so tests that patch the
# `requests` module attribute don't break exception handling
_HTTP_ERROR = requests.exceptions.HTTPError

logger = logging.getLogger(__name__)

_STOP = object()

# 4xx responses that are worth retrying (timeout, rate limit)
_RETRYABLE_CLIENT_ERRORS = {408, 429}


class ControlCenterNotifier:
    """Delivers alerts (JSON + MP4 clip) to a control-center endpoint.

    Deliveries run on a background worker with a bounded queue and
    exponential-backoff retries, so a slow or unreachable control center
    never blocks the analysis pipelines and short outages lose nothing.
    Shutdown is interruptible: ``close()`` aborts in-flight backoff waits
    and reports how many queued alerts could not be delivered.

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
        self._stop_event = threading.Event()
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
        """Stop the worker, aborting any in-flight retry waits."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(_STOP)  # wake a worker blocked on get()
        except queue.Full:
            pass
        self._worker.join(timeout)
        pending = self._queue.qsize()
        if pending:
            logger.warning(
                "Control-center notifier stopped with %d undelivered alert(s); "
                "they remain in local storage for manual follow-up", pending,
            )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is _STOP:
                return
            alert, clip_path = item
            self._deliver(alert, clip_path)
            self._queue.task_done()

    def _deliver(self, alert: dict, clip_path: str) -> None:
        for attempt in range(1, self._max_retries + 1):
            if self._stop_event.is_set():
                logger.warning(
                    "Shutdown during delivery of alert %s — abandoning",
                    alert.get("id"),
                )
                return
            try:
                self._post(alert, clip_path)
                logger.info(
                    "Alert %s delivered to control center (%s)",
                    alert.get("id"), self._url,
                )
                return
            except _HTTP_ERROR as exc:
                status = exc.response.status_code if exc.response is not None else None
                if (
                    status is not None
                    and 400 <= status < 500
                    and status not in _RETRYABLE_CLIENT_ERRORS
                ):
                    logger.error(
                        "Control center rejected alert %s with HTTP %s — not retrying",
                        alert.get("id"), status,
                    )
                    return
                self._wait_before_retry(attempt, exc)
            except Exception as exc:
                self._wait_before_retry(attempt, exc)
        logger.error(
            "Giving up on alert %s after %d attempts",
            alert.get("id"), self._max_retries,
        )

    def _wait_before_retry(self, attempt: int, exc: Exception) -> None:
        if attempt >= self._max_retries:
            return  # no point sleeping after the final attempt
        delay = self._backoff_base * (2 ** (attempt - 1))
        logger.warning(
            "Control-center delivery failed (attempt %d/%d): %s — retrying in %.0fs",
            attempt, self._max_retries, exc, delay,
        )
        # Interruptible: close() aborts the wait immediately
        self._stop_event.wait(delay)

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
