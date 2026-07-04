"""Tests for the control-center HTTP notifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.alerts.notifier import ControlCenterNotifier


ALERT = {"id": 7, "camera_id": "cam0", "alert_type": "violence", "confidence": 0.9}


@pytest.fixture()
def notifier():
    n = ControlCenterNotifier(
        url="http://control-center/api/alerts",
        api_key="secret",
        max_retries=3,
        backoff_base=0.0,
    )
    yield n
    n.close(timeout=2)


class TestDelivery:
    @patch("src.alerts.notifier.requests")
    def test_multipart_delivery_with_clip(self, mock_requests, notifier, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake video")

        notifier._deliver(ALERT, str(clip))

        mock_requests.post.assert_called_once()
        kwargs = mock_requests.post.call_args.kwargs
        assert kwargs["headers"] == {"X-API-Key": "secret"}
        assert json.loads(kwargs["data"]["alert"])["id"] == 7
        name, fh, content_type = kwargs["files"]["clip"]
        assert name == "clip.mp4"
        assert content_type == "video/mp4"

    @patch("src.alerts.notifier.requests")
    def test_json_delivery_when_clip_missing(self, mock_requests, notifier):
        notifier._deliver(ALERT, "does/not/exist.mp4")

        kwargs = mock_requests.post.call_args.kwargs
        assert kwargs["json"] == ALERT
        assert "files" not in kwargs

    @patch("src.alerts.notifier.requests")
    def test_retries_until_success(self, mock_requests, notifier):
        ok = MagicMock()
        mock_requests.post.side_effect = [ConnectionError("down"), ok]

        notifier._deliver(ALERT, "missing.mp4")

        assert mock_requests.post.call_count == 2

    @patch("src.alerts.notifier.requests")
    def test_gives_up_after_max_retries(self, mock_requests, notifier):
        mock_requests.post.side_effect = ConnectionError("down")

        notifier._deliver(ALERT, "missing.mp4")  # must not raise

        assert mock_requests.post.call_count == 3

    @patch("src.alerts.notifier.requests")
    def test_server_error_triggers_retry(self, mock_requests, notifier):
        bad = MagicMock()
        bad.raise_for_status.side_effect = RuntimeError("500")
        ok = MagicMock()
        mock_requests.post.side_effect = [bad, ok]

        notifier._deliver(ALERT, "missing.mp4")

        assert mock_requests.post.call_count == 2

    @patch("src.alerts.notifier.requests")
    def test_4xx_rejection_not_retried(self, mock_requests, notifier):
        import requests as real_requests

        error = real_requests.exceptions.HTTPError(
            response=MagicMock(status_code=422)
        )
        bad = MagicMock()
        bad.raise_for_status.side_effect = error
        mock_requests.post.return_value = bad

        notifier._deliver(ALERT, "missing.mp4")

        assert mock_requests.post.call_count == 1  # rejected, not retried

    @patch("src.alerts.notifier.requests")
    def test_429_is_retried(self, mock_requests, notifier):
        import requests as real_requests

        error = real_requests.exceptions.HTTPError(
            response=MagicMock(status_code=429)
        )
        bad = MagicMock()
        bad.raise_for_status.side_effect = error
        ok = MagicMock()
        mock_requests.post.side_effect = [bad, ok]

        notifier._deliver(ALERT, "missing.mp4")

        assert mock_requests.post.call_count == 2


class TestQueueing:
    @patch("src.alerts.notifier.requests")
    def test_notify_delivers_via_worker(self, mock_requests):
        n = ControlCenterNotifier(url="http://cc/alerts", max_retries=1)
        try:
            n.notify(ALERT, "missing.mp4")
            n._queue.join()  # wait for the worker to finish the delivery
            mock_requests.post.assert_called_once()
        finally:
            n.close(timeout=2)

    def test_full_queue_drops_without_blocking(self):
        n = ControlCenterNotifier(url="http://cc/alerts", queue_size=1)
        try:
            # Stall the worker by filling the queue faster than it drains
            with patch.object(n, "_deliver"):
                for _ in range(50):
                    n.notify(ALERT, "missing.mp4")  # must never block or raise
        finally:
            n.close(timeout=2)

    @patch("src.alerts.notifier.requests")
    def test_close_interrupts_retry_backoff(self, mock_requests):
        """Shutdown must not wait out minutes of exponential backoff."""
        import time

        mock_requests.post.side_effect = ConnectionError("down")
        n = ControlCenterNotifier(
            url="http://cc/alerts", max_retries=5, backoff_base=30.0
        )
        n.notify(ALERT, "missing.mp4")
        time.sleep(0.2)  # let the worker enter the first backoff wait

        start = time.monotonic()
        n.close(timeout=5)
        assert time.monotonic() - start < 3
        assert not n._worker.is_alive()
