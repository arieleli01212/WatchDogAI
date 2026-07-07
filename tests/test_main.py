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
    @patch("main.PipelineManager")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_starts_manager_in_configured_mode(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_manager_cls,
        mock_create_app,
        mock_uvicorn,
    ):
        settings = Settings(
            cameras=(
                CameraConfig(id="cam-north", source=0),
                CameraConfig(id="cam-south", source=1),
            ),
            source_mode="recordings",
            recordings_dir="C:/footage",
            dashboard_port=8000,
        )
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()
        mock_create_app.return_value = MagicMock()
        mock_uvicorn.run = MagicMock()  # returns immediately -> main() shuts down

        manager = mock_manager_cls.return_value
        manager.available_modes.return_value = ["live", "recordings"]

        from main import main
        main()

        manager.start.assert_called_once_with("recordings")
        manager.stop.assert_called_once()  # shutdown tears the pipelines down

        # Shared components created once
        mock_detector_cls.assert_called_once()
        mock_alert_manager_cls.assert_called_once_with(settings)
        mock_create_app.assert_called_once_with(settings)
        mock_uvicorn.run.assert_called_once()

    @patch("main.uvicorn")
    @patch("main.create_app")
    @patch("main.PipelineManager")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_falls_back_to_live_when_mode_unavailable(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_manager_cls,
        mock_create_app,
        mock_uvicorn,
    ):
        # recordings requested but no RECORDINGS_DIR configured
        settings = Settings(
            cameras=(CameraConfig(id="cam0", source=0),),
            source_mode="recordings",
            recordings_dir="",
        )
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()
        mock_create_app.return_value = MagicMock()
        mock_uvicorn.run = MagicMock()

        manager = mock_manager_cls.return_value
        manager.available_modes.return_value = ["live"]

        from main import main
        main()

        manager.start.assert_called_once_with("live")

    @patch("main.TelemetryLoop")
    @patch("main.MqttGatewayClient")
    @patch("main.ControlCenterNotifier")
    @patch("main.uvicorn")
    @patch("main.create_app")
    @patch("main.PipelineManager")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_wires_outbound_notifiers(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_manager_cls,
        mock_create_app,
        mock_uvicorn,
        mock_notifier_cls,
        mock_gateway_cls,
        mock_telemetry_cls,
    ):
        settings = Settings(
            cameras=(CameraConfig(id="cam0", source=0),),
            control_center_url="http://control-center/api/alerts",
            control_center_api_key="key",
            mqtt_host="gateway.campus",
        )
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()
        mock_create_app.return_value = MagicMock()
        mock_uvicorn.run = MagicMock()
        mock_manager_cls.return_value.available_modes.return_value = ["live"]

        from main import main
        main()

        mock_notifier_cls.assert_called_once_with(
            url="http://control-center/api/alerts", api_key="key"
        )
        mock_gateway_cls.assert_called_once()
        manager = mock_alert_manager_cls.return_value
        assert manager.add_notifier.call_count == 2

        mock_telemetry_cls.assert_called_once()
        mock_telemetry_cls.return_value.start.assert_called_once()

        # Shutdown releases the outbound channels
        mock_notifier_cls.return_value.close.assert_called_once()
        mock_gateway_cls.return_value.close.assert_called_once()

    @patch("main.uvicorn")
    @patch("main.create_app")
    @patch("main.PipelineManager")
    @patch("main.AlertManager")
    @patch("main.ViolenceDetector")
    @patch("main.setup_logging")
    @patch("main.get_settings")
    def test_main_wires_manager_to_app_state(
        self,
        mock_get_settings,
        mock_setup_logging,
        mock_detector_cls,
        mock_alert_manager_cls,
        mock_manager_cls,
        mock_create_app,
        mock_uvicorn,
    ):
        settings = Settings(cameras=(CameraConfig(id="cam0", source=0),))
        mock_get_settings.return_value = settings
        mock_setup_logging.return_value = MagicMock()

        app = MagicMock()
        mock_create_app.return_value = app
        mock_uvicorn.run = MagicMock()
        mock_manager_cls.return_value.available_modes.return_value = ["live"]

        from main import main
        main()

        # The manager gets the app's shared registries so the dashboard and
        # telemetry loop track pipeline swaps, and is exposed for the routes
        kwargs = mock_manager_cls.call_args.kwargs
        assert kwargs["cameras_registry"] is app.state.cameras
        assert kwargs["status_registry"] is app.state.camera_status
        assert app.state.pipeline_manager is mock_manager_cls.return_value
        assert app.state.alert_manager is mock_alert_manager_cls.return_value
