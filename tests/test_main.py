"""Tests for the main entry point module."""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import Settings
from main import setup_logging, detection_loop


class TestSetupLogging:
    """setup_logging should configure the root logger correctly."""

    def test_setup_logging_creates_logger(self, tmp_path):
        """setup_logging returns a logger named 'watchdog'."""
        settings = Settings(
            log_dir=str(tmp_path),
            log_level="DEBUG",
        )
        # Clear existing handlers to avoid interference
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            logger = setup_logging(settings)
            assert logger.name == "watchdog"
            assert (tmp_path / "watchdog.log").exists()
        finally:
            # Restore original state and close handlers we created
            for h in root.handlers[:]:
                h.close()
            root.handlers = original_handlers

    def test_setup_logging_respects_log_level(self, tmp_path):
        """The effective log level should match settings.log_level."""
        settings = Settings(
            log_dir=str(tmp_path),
            log_level="WARNING",
        )
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging(settings)
            assert logging.getLogger().level == logging.WARNING
        finally:
            for h in root.handlers[:]:
                h.close()
            root.handlers = original_handlers

    def test_setup_logging_creates_log_dir(self, tmp_path):
        """setup_logging should create the log directory if it doesn't exist."""
        log_dir = tmp_path / "nested" / "logs"
        settings = Settings(
            log_dir=str(log_dir),
            log_level="INFO",
        )
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging(settings)
            assert log_dir.exists()
        finally:
            for h in root.handlers[:]:
                h.close()
            root.handlers = original_handlers


class TestMainComponentsInitialize:
    """Verify main() initializes all components with correct settings."""

    @patch("main.uvicorn")
    @patch("main.create_app")
    @patch("main.ClipRecorder")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.Camera")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_components_initialize(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_camera_cls,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_clip_recorder_cls,
        mock_create_app,
        mock_uvicorn,
    ):
        """main() should create Camera, ViolenceDetector, AlertManager, ClipRecorder."""
        settings = Settings(
            camera_source=0,
            clip_length=90,
            dashboard_port=8000,
        )
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()

        # Keep the worker threads harmless: no frames, no video source
        camera = mock_camera_cls.return_value
        camera.read_frame.return_value = (False, None)
        camera.get_latest_frame.return_value = None
        camera.fps = 30.0
        camera._source = 0

        mock_create_app.return_value = MagicMock()
        mock_uvicorn.run = MagicMock()  # returns immediately -> main() shuts down

        from main import main
        main()

        mock_camera_cls.assert_called_once_with(
            source=settings.camera_source,
            clip_length=settings.clip_length,
        )
        mock_detector_cls.assert_called_once()
        mock_alert_manager_cls.assert_called_once_with(settings)
        mock_clip_recorder_cls.assert_called_once()
        mock_create_app.assert_called_once()
        mock_uvicorn.run.assert_called_once()
        camera.release.assert_called_once()


def _run_detection_loop(camera, detector, clip_recorder, settings, duration=0.4):
    """Run detection_loop in a thread for a short period, then stop it."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=detection_loop,
        args=(camera, detector, clip_recorder, settings, MagicMock(), stop_event),
        daemon=True,
    )
    thread.start()
    time.sleep(duration)
    stop_event.set()
    thread.join(timeout=3)
    return thread


class TestDetectionLoop:
    """detection_loop should classify frames and signal the clip recorder."""

    def test_confirmed_violence_after_consecutive_hits(self):
        """After N consecutive high-confidence detections, on_detection(True) fires."""
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.get_latest_frame.return_value = fake_frame

        detector = MagicMock()
        detector.predict_frame.return_value = ("violence", 0.95)

        clip_recorder = MagicMock()
        settings = Settings(confidence_threshold=0.85, consecutive_hits=3)

        _run_detection_loop(camera, detector, clip_recorder, settings)

        assert detector.predict_frame.call_count >= 3
        confirmed = [c.args[0] for c in clip_recorder.on_detection.call_args_list]
        # First two iterations are unconfirmed, then the streak confirms
        assert confirmed[:2] == [False, False]
        assert True in confirmed

    def test_normal_frames_never_confirm(self):
        """Normal predictions should never signal confirmed violence."""
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.get_latest_frame.return_value = fake_frame

        detector = MagicMock()
        detector.predict_frame.return_value = ("normal", 0.98)

        clip_recorder = MagicMock()
        settings = Settings(confidence_threshold=0.85, consecutive_hits=3)

        _run_detection_loop(camera, detector, clip_recorder, settings)

        assert clip_recorder.on_detection.called
        confirmed = [c.args[0] for c in clip_recorder.on_detection.call_args_list]
        assert True not in confirmed

    def test_low_confidence_violence_never_confirms(self):
        """Violence below the confidence threshold should not build a streak."""
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.get_latest_frame.return_value = fake_frame

        detector = MagicMock()
        detector.predict_frame.return_value = ("violence", 0.50)

        clip_recorder = MagicMock()
        settings = Settings(confidence_threshold=0.85, consecutive_hits=3)

        _run_detection_loop(camera, detector, clip_recorder, settings)

        confirmed = [c.args[0] for c in clip_recorder.on_detection.call_args_list]
        assert True not in confirmed

    def test_no_frames_skips_inference(self):
        """If the camera has no frames yet, predict_frame is never called."""
        camera = MagicMock()
        camera.get_latest_frame.return_value = None

        detector = MagicMock()
        clip_recorder = MagicMock()
        settings = Settings()

        _run_detection_loop(camera, detector, clip_recorder, settings, duration=0.2)

        detector.predict_frame.assert_not_called()
        clip_recorder.on_detection.assert_not_called()

    def test_detection_loop_updates_app_state(self):
        """When _app_state is set, detector_status should be updated."""
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        camera = MagicMock()
        camera.get_latest_frame.return_value = fake_frame

        detector = MagicMock()
        detector.predict_frame.return_value = ("violence", 0.92)

        clip_recorder = MagicMock()
        settings = Settings(confidence_threshold=0.85, consecutive_hits=3)

        app_state = MagicMock()
        detection_loop._app_state = app_state
        try:
            _run_detection_loop(camera, detector, clip_recorder, settings)
            status = app_state.detector_status
            assert status["label"] in ("violence", "normal")
            assert "violence_score" in status
            assert "streak" in status
        finally:
            detection_loop._app_state = None

    def test_detection_loop_stops_on_event(self):
        """Setting the stop_event should cause the loop to exit."""
        camera = MagicMock()
        camera.get_latest_frame.return_value = None

        thread = _run_detection_loop(
            camera, MagicMock(), MagicMock(), Settings(), duration=0.2
        )
        assert not thread.is_alive()
