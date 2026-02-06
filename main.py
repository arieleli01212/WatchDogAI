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


def detection_loop(camera, detector, alert_manager, settings, logger, stop_event):
    """Main detection loop running in a background thread."""
    logger.info("Detection loop started")
    inference_interval = settings.clip_length  # Process every clip_length frames
    frame_count = 0

    while not stop_event.is_set():
        success, frame = camera.read_frame()
        if not success:
            logger.warning("Failed to read frame, retrying...")
            time.sleep(0.1)
            continue

        camera.add_frame(frame)
        frame_count += 1

        # Run inference every clip_length frames when buffer is full
        if frame_count % inference_interval == 0:
            clip = camera.get_clip()
            if clip is not None:
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
                alert_manager.on_detection(label, confidence, frame)

        # Small sleep to control frame rate
        time.sleep(0.01)

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
    detector = ViolenceDetector(model_path=settings.model_path)
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

    # Start detection loop in background thread
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
        detection_thread.join(timeout=5)
        camera.release()
        logger.info("WatchDogAI stopped")


if __name__ == "__main__":
    main()
