"""Tests for the per-camera CameraPipeline."""

from __future__ import annotations

import itertools
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import CameraConfig, Settings
from src.pipeline import CameraPipeline


FAKE_FRAME = np.zeros((32, 32, 3), dtype=np.uint8)


@pytest.fixture()
def make_pipeline(tmp_path):
    """Factory that builds a CameraPipeline with mocked camera, recorder, and tracker."""

    def _make(settings: Settings | None = None, detector=None):
        settings = settings or Settings(
            confidence_threshold=0.85,
            consecutive_hits=3,
            clip_dir=str(tmp_path / "clips"),
            db_path=str(tmp_path / "db.sqlite"),
        )
        detector = detector or MagicMock()
        stop_event = threading.Event()
        with patch("src.pipeline.Camera") as mock_camera_cls, \
             patch("src.pipeline.ClipRecorder"), \
             patch("src.pipeline.ObjectTracker") as mock_tracker_cls, \
             patch("src.pipeline.BehaviorAnalyzer") as mock_behavior_cls:
            camera = mock_camera_cls.return_value
            camera.fps = 30.0
            tracker = mock_tracker_cls.return_value
            tracker.update.return_value = []
            tracker.counts = {
                "people": 0, "vehicles": 0, "unique_people": 0, "unique_vehicles": 0,
            }
            mock_behavior_cls.return_value.update.return_value = []
            pipeline = CameraPipeline(
                config=CameraConfig(id="cam-test", source=0, name="Test"),
                settings=settings,
                detector=detector,
                alert_manager=MagicMock(),
                status_registry={},
                stop_event=stop_event,
            )
        return pipeline, camera, pipeline.clip_recorder, stop_event

    return _make


def _run_analysis(pipeline, stop_event, duration=0.3):
    thread = threading.Thread(target=pipeline._analysis_loop, daemon=True)
    thread.start()
    time.sleep(duration)
    stop_event.set()
    thread.join(timeout=3)
    return thread


class TestPipelineConstruction:
    """Pipeline should build its camera from the camera config."""

    def test_camera_built_from_config(self, tmp_path):
        settings = Settings(
            clip_dir=str(tmp_path / "clips"),
            db_path=str(tmp_path / "db.sqlite"),
        )
        stop_event = threading.Event()
        with patch("src.pipeline.Camera") as mock_camera_cls, \
             patch("src.pipeline.ClipRecorder") as mock_recorder_cls, \
             patch("src.pipeline.ObjectTracker"), \
             patch("src.pipeline.BehaviorAnalyzer"):
            mock_camera_cls.return_value.fps = 25.0
            config = CameraConfig(
                id="cam-north", source="rtsp://x/stream", name="North",
                width=1280, height=720, fps=15,
            )
            CameraPipeline(
                config=config,
                settings=settings,
                detector=MagicMock(),
                alert_manager=MagicMock(),
                status_registry={},
                stop_event=stop_event,
            )
            mock_camera_cls.assert_called_once_with(
                source="rtsp://x/stream",
                camera_id="cam-north",
                name="North",
                width=1280,
                height=720,
                target_fps=15,
            )
            _, recorder_kwargs = mock_recorder_cls.call_args
            assert recorder_kwargs["camera_id"] == "cam-north"
            assert recorder_kwargs["fps"] == 25.0


class TestAnalysisLoop:
    """The analysis loop classifies distinct frames with temporal smoothing."""

    def test_same_frame_never_reclassified(self, make_pipeline):
        """Sequence gating: an unchanged frame is classified at most once."""
        pipeline, camera, recorder, stop_event = make_pipeline()
        camera.get_latest_frame_with_seq.return_value = (FAKE_FRAME, 1)
        pipeline._detector.predict_frame.return_value = ("violence", 0.99)

        _run_analysis(pipeline, stop_event)

        assert pipeline._detector.predict_frame.call_count == 1

    def test_streak_confirms_after_consecutive_distinct_frames(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("violence", 0.95)

        _run_analysis(pipeline, stop_event)

        confirmed = [c.args[0] for c in recorder.on_detection.call_args_list]
        assert len(confirmed) >= 3
        # First two distinct frames are unconfirmed, the third confirms
        assert confirmed[:3] == [False, False, True]

    def test_normal_frames_never_confirm(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("normal", 0.97)

        _run_analysis(pipeline, stop_event)

        confirmed = [c.args[0] for c in recorder.on_detection.call_args_list]
        assert confirmed and True not in confirmed

    def test_low_confidence_violence_never_confirms(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("violence", 0.50)

        _run_analysis(pipeline, stop_event)

        confirmed = [c.args[0] for c in recorder.on_detection.call_args_list]
        assert True not in confirmed

    def test_status_registry_updated_per_camera(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("violence", 0.92)

        _run_analysis(pipeline, stop_event)

        status = pipeline._status["cam-test"]
        assert status["label"] in ("violence", "normal")
        assert status["violence_score"] == pytest.approx(0.92)
        assert "streak" in status and "last_update" in status

    def test_no_frames_skips_inference(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        camera.get_latest_frame_with_seq.return_value = (None, 0)

        _run_analysis(pipeline, stop_event, duration=0.15)

        pipeline._detector.predict_frame.assert_not_called()
        recorder.on_detection.assert_not_called()

    def test_recorder_ticks_while_frames_stalled(self, make_pipeline):
        """A stalled camera must still let an overdue recording finalize."""
        pipeline, camera, recorder, stop_event = make_pipeline()
        camera.get_latest_frame_with_seq.return_value = (None, 0)

        _run_analysis(pipeline, stop_event, duration=0.15)

        assert recorder.tick.called


class TestObjectTracking:
    """People/vehicle tracking runs inside the analysis loop."""

    def test_tracker_results_published_in_status(self, make_pipeline):
        from src.detector.objects import TrackedObject

        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("normal", 0.9)
        pipeline._tracker.update.return_value = [
            TrackedObject(
                track_id=7, category="person", label="person",
                confidence=0.88, box=(1.0, 2.0, 3.0, 4.0),
            )
        ]
        pipeline._tracker.counts = {
            "people": 1, "vehicles": 0, "unique_people": 1, "unique_vehicles": 0,
        }

        _run_analysis(pipeline, stop_event)

        status = pipeline._status["cam-test"]
        assert status["counts"]["people"] == 1
        assert status["objects"][0]["track_id"] == 7
        assert status["objects"][0]["category"] == "person"
        assert status["objects"][0]["box"] == [1.0, 2.0, 3.0, 4.0]

    def test_tracker_disabled_by_settings(self, tmp_path):
        settings = Settings(
            object_detection_enabled=False,
            clip_dir=str(tmp_path / "clips"),
            db_path=str(tmp_path / "db.sqlite"),
        )
        stop_event = threading.Event()
        with patch("src.pipeline.Camera") as mock_camera_cls, \
             patch("src.pipeline.ClipRecorder"), \
             patch("src.pipeline.ObjectTracker") as mock_tracker_cls:
            mock_camera_cls.return_value.fps = 30.0
            pipeline = CameraPipeline(
                config=CameraConfig(id="cam-test", source=0),
                settings=settings,
                detector=MagicMock(),
                alert_manager=MagicMock(),
                status_registry={},
                stop_event=stop_event,
            )
        mock_tracker_cls.assert_not_called()
        assert pipeline._tracker is None

    def test_tracker_construction_failure_is_not_fatal(self, tmp_path):
        settings = Settings(
            clip_dir=str(tmp_path / "clips"),
            db_path=str(tmp_path / "db.sqlite"),
        )
        stop_event = threading.Event()
        with patch("src.pipeline.Camera") as mock_camera_cls, \
             patch("src.pipeline.ClipRecorder"), \
             patch("src.pipeline.ObjectTracker", side_effect=RuntimeError("no yolo")):
            mock_camera_cls.return_value.fps = 30.0
            pipeline = CameraPipeline(
                config=CameraConfig(id="cam-test", source=0),
                settings=settings,
                detector=MagicMock(),
                alert_manager=MagicMock(),
                status_registry={},
                stop_event=stop_event,
            )
        assert pipeline._tracker is None

    def test_tracker_failure_mid_run_disables_tracking(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("normal", 0.9)
        pipeline._tracker.update.side_effect = RuntimeError("boom")

        _run_analysis(pipeline, stop_event)

        # Loop kept running (status written) with tracking disabled
        assert pipeline._tracker is None
        status = pipeline._status["cam-test"]
        assert status["objects"] == []
        assert status["counts"] == {}


class TestBehaviorIntegration:
    """Behavior events flow from the analyzer into typed alerts."""

    LOITER_EVENT = {
        "type": "loitering", "track_id": 4, "category": "person",
        "score": 0.8, "details": "present 65s",
    }

    def test_behavior_event_triggers_typed_alert(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("normal", 0.9)
        pipeline._behavior.update.return_value = [self.LOITER_EVENT]

        _run_analysis(pipeline, stop_event)

        typed_calls = [
            c for c in recorder.on_detection.call_args_list
            if c.kwargs.get("alert_type") == "loitering"
        ]
        assert typed_calls
        assert typed_calls[0].args == (True, 0.8)

    def test_violence_takes_precedence_over_behavior(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("violence", 0.95)
        pipeline._behavior.update.return_value = [self.LOITER_EVENT]

        _run_analysis(pipeline, stop_event)

        confirmed = [
            c for c in recorder.on_detection.call_args_list if c.args[0] is True
        ]
        assert confirmed
        # Once the violence streak confirms, the alert type must be violence
        assert all(c.kwargs.get("alert_type") == "violence" for c in confirmed[2:])

    def test_behavior_events_published_in_status(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("normal", 0.9)
        pipeline._behavior.update.return_value = [self.LOITER_EVENT]

        _run_analysis(pipeline, stop_event)

        assert pipeline._status["cam-test"]["behavior_events"] == [self.LOITER_EVENT]

    def test_behavior_failure_disables_analytics(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        seq = itertools.count(1)
        camera.get_latest_frame_with_seq.side_effect = lambda: (FAKE_FRAME, next(seq))
        pipeline._detector.predict_frame.return_value = ("normal", 0.9)
        pipeline._behavior.update.side_effect = RuntimeError("boom")

        _run_analysis(pipeline, stop_event)

        assert pipeline._behavior is None
        assert pipeline._status["cam-test"]["behavior_events"] == []


class TestCaptureLoop:
    """The capture loop feeds frames to the buffer and clip recorder."""

    def test_frames_fanned_out(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()

        reads = itertools.count()

        def read_side_effect():
            if next(reads) >= 5:
                stop_event.set()
                return None
            return FAKE_FRAME

        camera.read.side_effect = read_side_effect

        thread = threading.Thread(target=pipeline._capture_loop, daemon=True)
        thread.start()
        thread.join(timeout=3)

        assert camera.add_frame.call_count == 5
        assert recorder.add_frame.call_count == 5

    def test_none_frames_skipped(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()

        reads = itertools.count()

        def read_side_effect():
            n = next(reads)
            if n >= 4:
                stop_event.set()
                return None
            return None if n % 2 else FAKE_FRAME

        camera.read.side_effect = read_side_effect

        thread = threading.Thread(target=pipeline._capture_loop, daemon=True)
        thread.start()
        thread.join(timeout=3)

        assert camera.add_frame.call_count == 2


class TestLifecycle:
    """start/join/release should manage both worker threads."""

    def test_start_and_join(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        camera.read.return_value = None
        camera.get_latest_frame_with_seq.return_value = (None, 0)

        pipeline.start()
        assert len(pipeline._threads) == 2
        stop_event.set()
        pipeline.join(timeout=3)
        assert all(not t.is_alive() for t in pipeline._threads)

    def test_release_releases_camera(self, make_pipeline):
        pipeline, camera, recorder, stop_event = make_pipeline()
        pipeline.release()
        camera.release.assert_called_once()
