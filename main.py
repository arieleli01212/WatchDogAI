"""WatchDogAI - AI-powered smart security system entry point."""

import logging
import os
import signal
import threading

import uvicorn

from src.config import get_settings
from src.detector.model import ViolenceDetector
from src.alerts.manager import AlertManager
from src.alerts.notifier import ControlCenterNotifier
from src.mqtt.client import MqttGatewayClient, TelemetryLoop
from src.pipeline import CameraPipeline
from src.dashboard.app import create_app


def setup_logging(settings):
    """Configure basic logging."""
    log_dir = settings.log_dir
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/watchdog.log"),
        ],
    )
    return logging.getLogger("watchdog")


def main():
    settings = get_settings()
    logger = setup_logging(settings)

    logger.info("WatchDogAI starting...")
    logger.info(
        "Cameras: %s",
        ", ".join(f"{c.id} ({c.source})" for c in settings.cameras),
    )
    logger.info(f"Confidence threshold: {settings.confidence_threshold}")
    logger.info(f"Dashboard port: {settings.dashboard_port}")

    # Shared components: one model instance and one alert manager serve all cameras
    detector = ViolenceDetector()
    alert_manager = AlertManager(settings)

    # Outbound alerting: HTTP push to the municipal control center
    notifier = None
    if settings.control_center_url:
        notifier = ControlCenterNotifier(
            url=settings.control_center_url,
            api_key=settings.control_center_api_key,
        )
        alert_manager.add_notifier(notifier)
        logger.info("Control-center notifier -> %s", settings.control_center_url)

    # Outbound alerting + telemetry over the LoRa gateway (MQTT)
    gateway = None
    if settings.mqtt_host:
        try:
            gateway = MqttGatewayClient(
                host=settings.mqtt_host,
                port=settings.mqtt_port,
                username=settings.mqtt_username,
                password=settings.mqtt_password,
                base_topic=settings.mqtt_base_topic,
            )
            alert_manager.add_notifier(gateway)
        except Exception:
            logger.exception("MQTT gateway unavailable, continuing without it")

    # Create dashboard app and wire components
    app = create_app(settings)
    app.state.alert_manager = alert_manager

    # Setup graceful shutdown
    stop_event = threading.Event()

    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # One pipeline (capture + analysis threads) per configured camera
    pipelines = []
    for camera_config in settings.cameras:
        pipeline = CameraPipeline(
            config=camera_config,
            settings=settings,
            detector=detector,
            alert_manager=alert_manager,
            status_registry=app.state.camera_status,
            stop_event=stop_event,
        )
        app.state.cameras[camera_config.id] = pipeline.camera
        pipelines.append(pipeline)

    for pipeline in pipelines:
        pipeline.start()
        logger.info("Pipeline started for camera %s", pipeline.config.id)

    # Periodic camera-health telemetry over the gateway
    if gateway is not None:
        TelemetryLoop(
            gateway=gateway,
            cameras=app.state.cameras,
            status_registry=app.state.camera_status,
            interval=settings.telemetry_interval,
            stop_event=stop_event,
            health_max_age=settings.camera_health_max_age,
        ).start()

    # Start dashboard (blocks until shutdown)
    try:
        uvicorn.run(app, host="0.0.0.0", port=settings.dashboard_port, log_level="info")
    finally:
        logger.info("Shutting down...")
        stop_event.set()
        for pipeline in pipelines:
            pipeline.join(timeout=5)
            pipeline.release()
        if notifier is not None:
            notifier.close()
        if gateway is not None:
            gateway.close()
        logger.info("WatchDogAI stopped")


if __name__ == "__main__":
    main()
