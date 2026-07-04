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
        max_clip_seconds=0.5,
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

        with patch("src.alerts.clip_recorder.cv2.VideoWriter") as mock_writer_cls:
            mock_writer_cls.return_value.isOpened.return_value = False
            recorder.add_frame(FRAME)
            recorder.on_detection(True, 0.9)
            recorder.add_frame(FRAME)
            recorder.on_detection(False, 0.1)  # post=0 -> SAVING with broken writer
            recorder.add_frame(FRAME)
            _wait_for_idle(recorder)

        alert_manager.on_clip_saved.assert_not_called()
        # Recorder must recover to IDLE so later events still record
        assert recorder._state is _State.IDLE


class TestRobustness:
    """Review findings: stalled cameras, sustained events, back-to-back events."""

    def test_tick_finalizes_recording_when_frames_stop(self, tmp_path):
        """Camera dies right after the incident: tick() must still save the clip."""
        settings = Settings(
            clip_dir=str(tmp_path / "clips2"),
            db_path=str(tmp_path / "db2.sqlite"),
            pre_event_seconds=0.1,
            post_event_seconds=0.05,
        )
        recorder = ClipRecorder(
            settings=settings, alert_manager=MagicMock(), fps=10.0,
            camera_id="cam-test",
        )
        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)  # deadline armed but not yet due
        assert recorder._state is _State.RECORDING

        # No frames ever arrive again; only the analysis loop's tick runs
        time.sleep(0.1)
        recorder.tick()
        _wait_for_idle(recorder)

        recorder._alert_manager.on_clip_saved.assert_called_once()

    def test_max_clip_length_chunks_sustained_event(self, settings):
        """A never-ending event must not accumulate frames unboundedly."""
        recorder = ClipRecorder(
            settings=settings, alert_manager=MagicMock(), fps=10.0,
            camera_id="cam-test",
        )
        # fps=10, max_clip_seconds from settings fixture below (0.5s -> 5 frames)
        recorder.on_detection(True, 0.9)
        for _ in range(20):
            recorder.add_frame(FRAME)
        _wait_for_idle(recorder)

        # The recording was force-chunked instead of growing to 20 frames
        assert recorder._alert_manager.on_clip_saved.called
        assert len(recorder._rec_frames) < 20

    def test_pre_buffer_keeps_filling_during_saving(self, recorder: ClipRecorder):
        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        recorder.add_frame(FRAME)
        with recorder._lock:
            recorder._transition_to_saving()
        # Frames arriving while the writer runs must land in the pre-buffer
        recorder.add_frame(FRAME)
        recorder.add_frame(FRAME)
        assert len(recorder._pre_buffer) >= 1

    def test_event_during_saving_starts_new_recording(self, recorder: ClipRecorder):
        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        recorder.add_frame(FRAME)
        with recorder._lock:
            recorder._state = _State.SAVING  # simulate an in-flight write

        recorder.on_detection(True, 0.95)
        assert recorder._state is _State.RECORDING

    def test_violence_upgrades_alert_type_mid_recording(self, recorder: ClipRecorder):
        alert_manager = recorder._alert_manager

        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.7, alert_type="loitering")
        recorder.add_frame(FRAME)
        # Violence breaks out during the behavior-event recording
        recorder.on_detection(True, 0.9, alert_type="violence")
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)
        recorder.add_frame(FRAME)
        recorder.tick()
        _wait_for_idle(recorder)

        assert alert_manager.on_clip_saved.call_args.kwargs["alert_type"] == "violence"

    def test_unsafe_camera_id_sanitized_in_filename(self, settings):
        recorder = ClipRecorder(
            settings=settings, alert_manager=MagicMock(), fps=10.0,
            camera_id="cam/../evil:id",
        )
        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)
        recorder.tick()
        _wait_for_idle(recorder)

        clip_path = recorder._alert_manager.on_clip_saved.call_args.kwargs["clip_path"]
        from pathlib import Path as P
        name = P(clip_path).name
        assert "/" not in name and ":" not in name and ".." not in name
        assert P(clip_path).is_file()
