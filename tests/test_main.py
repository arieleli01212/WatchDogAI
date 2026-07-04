"""Tests for the main entry point module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from src.config import CameraConfig, Settings
from main import setup_logging


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


class TestMain:
    """Verify main() builds one pipeline per camera and wires shared components."""

    @patch("main.uvicorn")
    @patch("main.create_app")
    @patch("main.CameraPipeline")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_builds_pipeline_per_camera(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_pipeline_cls,
        mock_create_app,
        mock_uvicorn,
    ):
        settings = Settings(
            cameras=(
                CameraConfig(id="cam-north", source=0),
                CameraConfig(id="cam-south", source=1),
            ),
            dashboard_port=8000,
        )
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()
        mock_create_app.return_value = MagicMock()
        mock_uvicorn.run = MagicMock()  # returns immediately -> main() shuts down

        from main import main
        main()

        # One pipeline per configured camera, all started and cleaned up
        assert mock_pipeline_cls.call_count == 2
        camera_ids = [
            call.kwargs["config"].id for call in mock_pipeline_cls.call_args_list
        ]
        assert camera_ids == ["cam-north", "cam-south"]

        pipeline = mock_pipeline_cls.return_value
        assert pipeline.start.call_count == 2
        assert pipeline.join.call_count == 2
        assert pipeline.release.call_count == 2

        # Shared components created once
        mock_detector_cls.assert_called_once()
        mock_alert_manager_cls.assert_called_once_with(settings)
        mock_create_app.assert_called_once_with(settings)
        mock_uvicorn.run.assert_called_once()

    @patch("main.uvicorn")
    @patch("main.create_app")
    @patch("main.CameraPipeline")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_registers_cameras_on_app_state(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_pipeline_cls,
        mock_create_app,
        mock_uvicorn,
    ):
        settings = Settings(cameras=(CameraConfig(id="cam0", source=0),))
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()

        app = MagicMock()
        app.state.cameras = {}
        mock_create_app.return_value = app
        mock_uvicorn.run = MagicMock()

        from main import main
        main()

        assert "cam0" in app.state.cameras
        assert app.state.alert_manager is mock_alert_manager_cls.return_value
