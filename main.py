"""WatchDogAI - AI-powered smart security system entry point."""

import logging
import os
import signal
import threading

import uvicorn

from src.config import get_settings
from src.detector.model import ViolenceDetector
from src.alerts.manager import AlertManager
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

    # Start dashboard (blocks until shutdown)
    try:
        uvicorn.run(app, host="0.0.0.0", port=settings.dashboard_port, log_level="info")
    finally:
        logger.info("Shutting down...")
        stop_event.set()
        for pipeline in pipelines:
            pipeline.join(timeout=5)
            pipeline.release()
        logger.info("WatchDogAI stopped")


if __name__ == "__main__":
    main()
