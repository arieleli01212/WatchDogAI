"""Tests for the ClipRecorder state machine."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.alerts.clip_recorder import ClipRecorder, _State
from src.config import Settings


FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        clip_dir=str(tmp_path / "clips"),
        db_path=str(tmp_path / "db.sqlite"),
        pre_event_seconds=0.1,
        post_event_seconds=0.0,
    )


@pytest.fixture()
def recorder(settings: Settings) -> ClipRecorder:
    return ClipRecorder(
        settings=settings,
        alert_manager=MagicMock(),
        fps=10.0,
        camera_id="cam-test",
    )


def _wait_for_idle(recorder: ClipRecorder, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with recorder._lock:
            if recorder._state is _State.IDLE:
                return
        time.sleep(0.01)
    raise TimeoutError("recorder did not return to IDLE")


class TestStateMachine:
    def test_starts_idle(self, recorder: ClipRecorder):
        assert recorder._state is _State.IDLE

    def test_event_starts_recording(self, recorder: ClipRecorder):
        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        assert recorder._state is _State.RECORDING

    def test_non_event_keeps_idle(self, recorder: ClipRecorder):
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.2)
        assert recorder._state is _State.IDLE

    def test_full_cycle_saves_clip_and_alerts(self, recorder: ClipRecorder, settings):
        alert_manager = recorder._alert_manager

        # Pre-event frames, then confirmed event, then event ends
        for _ in range(3):
            recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.93)
        for _ in range(5):
            recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)  # post_event_seconds=0 -> deadline now
        recorder.add_frame(FRAME)  # triggers the SAVING transition

        _wait_for_idle(recorder)

        alert_manager.on_clip_saved.assert_called_once()
        kwargs = alert_manager.on_clip_saved.call_args.kwargs
        assert kwargs["camera_id"] == "cam-test"
        assert kwargs["confidence"] == 0.93
        assert kwargs["alert_type"] == "violence"
        clip_file = Path(kwargs["clip_path"])
        assert clip_file.is_file()
        assert clip_file.stat().st_size > 0
        assert "cam-test" in clip_file.name

    def test_alert_type_propagates(self, recorder: ClipRecorder):
        alert_manager = recorder._alert_manager

        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.7, alert_type="abnormal_behavior")
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)
        recorder.add_frame(FRAME)

        _wait_for_idle(recorder)
        assert (
            alert_manager.on_clip_saved.call_args.kwargs["alert_type"]
            == "abnormal_behavior"
        )

    def test_failed_writer_skips_alert(self, recorder: ClipRecorder):
        alert_manager = recorder._alert_manager

        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)

        with patch("src.alerts.clip_recorder.cv2.VideoWriter") as mock_writer_cls:
            mock_writer_cls.return_value.isOpened.return_value = False
            recorder.add_frame(FRAME)  # triggers SAVING with the broken writer
            _wait_for_idle(recorder)

        alert_manager.on_clip_saved.assert_not_called()
        # Recorder must recover to IDLE so later events still record
        assert recorder._state is _State.IDLE
