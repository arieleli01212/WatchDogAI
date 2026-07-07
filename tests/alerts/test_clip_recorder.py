"""Tests for the ClipRecorder state machine."""

from __future__ import annotations

import shutil
import subprocess
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


class TestBrowserPlayableClips:
    """Saved clips must be H.264 — browsers render mp4v (MPEG-4 Part 2) as black."""

    @pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
        reason="ffmpeg/ffprobe not installed",
    )
    def test_saved_clip_is_h264_when_ffmpeg_available(self, recorder: ClipRecorder):
        alert_manager = recorder._alert_manager

        recorder.add_frame(FRAME)
        recorder.on_detection(True, 0.9)
        recorder.add_frame(FRAME)
        recorder.on_detection(False, 0.1)
        recorder.add_frame(FRAME)
        _wait_for_idle(recorder)

        clip_path = alert_manager.on_clip_saved.call_args.kwargs["clip_path"]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", clip_path],
            capture_output=True, text=True,
        )
        assert probe.stdout.strip() == "h264"
        # No leftover intermediate file next to the final clip
        assert list(Path(clip_path).parent.glob("*.tmp*")) == []

    def test_clip_saved_without_ffmpeg_falls_back_to_mp4v(self, recorder: ClipRecorder):
        alert_manager = recorder._alert_manager

        with patch("src.alerts.clip_recorder.shutil.which", return_value=None):
            recorder.add_frame(FRAME)
            recorder.on_detection(True, 0.9)
            recorder.add_frame(FRAME)
            recorder.on_detection(False, 0.1)
            recorder.add_frame(FRAME)
            _wait_for_idle(recorder)

        alert_manager.on_clip_saved.assert_called_once()
        clip_path = Path(alert_manager.on_clip_saved.call_args.kwargs["clip_path"])
        assert clip_path.is_file()
        assert clip_path.stat().st_size > 0
        assert list(clip_path.parent.glob("*.tmp*")) == []

    def test_direct_h264_tried_first_then_mp4v_fallback(self, recorder, tmp_path):
        """avc1 writers are probed before any mp4v fallback, and the probe result is cached."""
        import cv2 as real_cv2

        attempts: list[int] = []
        avc1 = real_cv2.VideoWriter_fourcc(*"avc1")
        mp4v = real_cv2.VideoWriter_fourcc(*"mp4v")

        class FakeWriter:
            """Simulates a system where only mp4v encoding works."""

            def __init__(self, path, *args):
                # Signatures: (path, fourcc, fps, size) or (path, api, fourcc, fps, size)
                self._path = path
                fourcc = args[1] if len(args) == 4 else args[0]
                attempts.append(int(fourcc))
                self._ok = int(fourcc) == mp4v

            def isOpened(self):
                return self._ok

            def write(self, frame):
                with open(self._path, "ab") as fh:
                    fh.write(b"x")

            def release(self):
                pass

        with patch("src.alerts.clip_recorder.cv2.VideoWriter", FakeWriter), \
             patch.object(recorder, "_reencode_h264", return_value=False):
            ok = recorder._encode_clip([FRAME, FRAME], tmp_path / "a.mp4")
            assert ok is True
            assert attempts[0] == avc1   # H.264 probed first
            assert attempts[-1] == mp4v  # fallback actually used

            attempts.clear()
            recorder._encode_clip([FRAME], tmp_path / "b.mp4")
            assert attempts == [mp4v]  # failed probe cached: no avc1 retries

    def test_direct_h264_success_is_cached(self, recorder, tmp_path):
        import cv2 as real_cv2

        attempts: list[int] = []
        avc1 = real_cv2.VideoWriter_fourcc(*"avc1")

        class AlwaysWorksWriter:
            def __init__(self, path, *args):
                self._path = path
                fourcc = args[1] if len(args) == 4 else args[0]
                attempts.append(int(fourcc))

            def isOpened(self):
                return True

            def write(self, frame):
                with open(self._path, "ab") as fh:
                    fh.write(b"x")

            def release(self):
                pass

        with patch("src.alerts.clip_recorder.cv2.VideoWriter", AlwaysWorksWriter):
            assert recorder._encode_clip([FRAME], tmp_path / "a.mp4") is True
            assert recorder._encode_clip([FRAME], tmp_path / "b.mp4") is True

        # First clip probes avc1 and succeeds; second reuses the cached combo
        assert attempts == [avc1, avc1]


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
