"""WatchDogAI - Real-time violence detection system."""

import logging
import threading
import time
import signal
import sys

import cv2
import uvicorn

from src.config import get_settings
from src.capture.camera import Camera
from src.detector.model import ViolenceDetector
from src.alerts.manager import AlertManager
from src.dashboard.app import create_app


def setup_logging(settings):
    """Configure basic logging."""
    log_dir = settings.log_dir
    # Ensure log directory exists
    import os
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


def capture_loop(camera, logger, stop_event):
    """Continuously read frames from the camera at full speed."""
    logger.info("Capture loop started")
    while not stop_event.is_set():
        success, frame = camera.read_frame()
        if not success:
            # For video files: loop back to the beginning
            if isinstance(camera._source, str):
                camera._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                logger.info("Video ended, looping...")
                continue
            time.sleep(0.03)
            continue
        camera.add_frame(frame)
        # Pace video files to their native FPS
        if isinstance(camera._source, str) and camera.fps > 0:
            time.sleep(1.0 / camera.fps)
    logger.info("Capture loop stopped")


def detection_loop(camera, detector, alert_manager, settings, logger, stop_event):
    """Run inference periodically on buffered clips."""
    logger.info("Detection loop started")

    while not stop_event.is_set():
        clip = camera.get_clip()
        if clip is None:
            time.sleep(0.1)
            continue

        label, confidence = detector.predict(clip)
        logger.debug(f"Detection: {label} ({confidence:.2f})")

        # Update dashboard status
        app_state = getattr(detection_loop, '_app_state', None)
        if app_state:
            app_state.detector_status = {
                "label": label,
                "confidence": confidence,
                "last_update": time.time(),
            }

        # Process alert
        latest_frame = camera.get_latest_frame()
        if latest_frame is not None:
            alert_manager.on_detection(label, confidence, latest_frame)

        # Wait before next inference to avoid hammering CPU
        time.sleep(0.5)

    logger.info("Detection loop stopped")


def main():
    settings = get_settings()
    logger = setup_logging(settings)

    logger.info("WatchDogAI starting...")
    logger.info(f"Camera source: {settings.camera_source}")
    logger.info(f"Confidence threshold: {settings.confidence_threshold}")
    logger.info(f"Dashboard port: {settings.dashboard_port}")

    # Initialize components
    camera = Camera(source=settings.camera_source, clip_length=settings.clip_length)
    detector = ViolenceDetector()
    alert_manager = AlertManager(settings)

    # Create dashboard app and wire components
    app = create_app()
    app.state.camera = camera
    app.state.alert_manager = alert_manager

    # Setup graceful shutdown
    stop_event = threading.Event()

    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received")
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Store app state reference for detection loop
    detection_loop._app_state = app.state

    # Start capture loop - reads frames continuously at full speed
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(camera, logger, stop_event),
        daemon=True,
    )
    capture_thread.start()
    logger.info("Capture thread started")

    # Start detection loop - runs inference periodically
    detection_thread = threading.Thread(
        target=detection_loop,
        args=(camera, detector, alert_manager, settings, logger, stop_event),
        daemon=True,
    )
    detection_thread.start()
    logger.info("Detection thread started")

    # Start dashboard (blocks until shutdown)
    try:
        uvicorn.run(app, host="0.0.0.0", port=settings.dashboard_port, log_level="info")
    finally:
        logger.info("Shutting down...")
        stop_event.set()
        capture_thread.join(timeout=5)
        detection_thread.join(timeout=5)
        camera.release()
        logger.info("WatchDogAI stopped")


if __name__ == "__main__":
    main()
