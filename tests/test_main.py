"""Tests for the main entry point module."""

from __future__ import annotations

import logging
import os
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

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
            logger = setup_logging(settings)
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
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.Camera")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_components_initialize(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_camera,
        mock_detector,
        mock_alert_manager,
        mock_create_app,
        mock_uvicorn,
    ):
        """main() should create Camera, ViolenceDetector, AlertManager with settings."""
        settings = Settings(
            camera_source=0,
            clip_length=16,
            model_path="models/test.pt",
            dashboard_port=8000,
        )
        mock_get_settings.return_value = settings
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger

        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        # uvicorn.run will be called; just let it return immediately
        mock_uvicorn.run = MagicMock()

        from main import main
        main()

        mock_camera.assert_called_once_with(
            source=settings.camera_source,
            clip_length=settings.clip_length,
        )
        mock_detector.assert_called_once_with(model_path=settings.model_path)
        mock_alert_manager.assert_called_once_with(settings)
        mock_create_app.assert_called_once()
        mock_uvicorn.run.assert_called_once()


class TestDetectionLoop:
    """detection_loop should read frames, run inference, and trigger alerts."""

    def test_detection_loop_processes_frames(self):
        """After clip_length frames, detector.predict should be called."""
        clip_length = 4
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        fake_clip = np.stack([fake_frame] * clip_length)

        mock_camera = MagicMock()
        # Return fake frames for clip_length frames, then stop
        call_count = 0

        def read_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= clip_length:
                return True, fake_frame
            return False, None

        mock_camera.read_frame.side_effect = read_side_effect
        mock_camera.get_clip.return_value = fake_clip

        mock_detector = MagicMock()
        mock_detector.predict.return_value = ("normal", 0.95)

        mock_alert_manager = MagicMock()

        settings = Settings(clip_length=clip_length)

        mock_logger = MagicMock()
        stop_event = threading.Event()

        # Run detection loop in a thread; it will stop after frames run out
        # and read_frame returns False (then it sleeps, we stop it)
        def run_loop():
            detection_loop(
                mock_camera, mock_detector, mock_alert_manager,
                settings, mock_logger, stop_event,
            )

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()

        # Wait a bit for the loop to process frames
        time.sleep(1.0)
        stop_event.set()
        thread.join(timeout=3)

        # Verify frames were read and added
        assert mock_camera.read_frame.call_count >= clip_length
        assert mock_camera.add_frame.call_count == clip_length

        # Verify inference was run
        mock_detector.predict.assert_called_once_with(fake_clip)

        # Verify alert processing was called
        mock_alert_manager.on_detection.assert_called_once_with(
            "normal", 0.95, fake_frame,
        )

    def test_detection_loop_skips_when_clip_not_ready(self):
        """If get_clip returns None, predict should not be called."""
        clip_length = 4
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)

        mock_camera = MagicMock()
        call_count = 0

        def read_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= clip_length:
                return True, fake_frame
            return False, None

        mock_camera.read_frame.side_effect = read_side_effect
        mock_camera.get_clip.return_value = None  # Buffer not full

        mock_detector = MagicMock()
        mock_alert_manager = MagicMock()
        settings = Settings(clip_length=clip_length)
        mock_logger = MagicMock()
        stop_event = threading.Event()

        thread = threading.Thread(
            target=detection_loop,
            args=(mock_camera, mock_detector, mock_alert_manager,
                  settings, mock_logger, stop_event),
            daemon=True,
        )
        thread.start()
        time.sleep(1.0)
        stop_event.set()
        thread.join(timeout=3)

        # predict should never have been called
        mock_detector.predict.assert_not_called()

    def test_detection_loop_updates_app_state(self):
        """When _app_state is set, detector_status should be updated."""
        clip_length = 4
        fake_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        fake_clip = np.stack([fake_frame] * clip_length)

        mock_camera = MagicMock()
        call_count = 0

        def read_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= clip_length:
                return True, fake_frame
            return False, None

        mock_camera.read_frame.side_effect = read_side_effect
        mock_camera.get_clip.return_value = fake_clip

        mock_detector = MagicMock()
        mock_detector.predict.return_value = ("violence", 0.92)

        mock_alert_manager = MagicMock()
        settings = Settings(clip_length=clip_length)
        mock_logger = MagicMock()
        stop_event = threading.Event()

        # Set up app state
        mock_app_state = MagicMock()
        detection_loop._app_state = mock_app_state

        thread = threading.Thread(
            target=detection_loop,
            args=(mock_camera, mock_detector, mock_alert_manager,
                  settings, mock_logger, stop_event),
            daemon=True,
        )
        thread.start()
        time.sleep(1.0)
        stop_event.set()
        thread.join(timeout=3)

        # Verify app state was updated
        assert mock_app_state.detector_status is not None

        # Clean up
        detection_loop._app_state = None

    def test_detection_loop_stops_on_event(self):
        """Setting the stop_event should cause the loop to exit."""
        mock_camera = MagicMock()
        mock_camera.read_frame.return_value = (False, None)

        mock_detector = MagicMock()
        mock_alert_manager = MagicMock()
        settings = Settings(clip_length=16)
        mock_logger = MagicMock()
        stop_event = threading.Event()

        thread = threading.Thread(
            target=detection_loop,
            args=(mock_camera, mock_detector, mock_alert_manager,
                  settings, mock_logger, stop_event),
            daemon=True,
        )
        thread.start()
        time.sleep(0.3)
        stop_event.set()
        thread.join(timeout=3)

        assert not thread.is_alive()
