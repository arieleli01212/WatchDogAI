"""WatchDogAI - Real-time violence detection system."""

import logging
import threading
import time
import signal

import cv2
import uvicorn

from src.config import get_settings
from src.capture.camera import Camera
from src.detector.model import ViolenceDetector
from src.alerts.manager import AlertManager
from src.alerts.clip_recorder import ClipRecorder
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


def capture_loop(camera, clip_recorder, logger, stop_event):
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
        clip_recorder.add_frame(frame)
        # Pace video files to their native FPS
        if isinstance(camera._source, str) and camera.fps > 0:
            time.sleep(1.0 / camera.fps)
    logger.info("Capture loop stopped")


def detection_loop(camera, detector, clip_recorder, settings, logger, stop_event):
    """Run inference on the latest frame for real-time detection.

    Uses temporal smoothing: only triggers an alert after N consecutive
    high-confidence violence detections. This eliminates false positives
    while keeping detection responsive.
    """
    logger.info("Detection loop started")
    consecutive_violence = 0
    required_hits = settings.consecutive_hits

    while not stop_event.is_set():
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        label, confidence = detector.predict_frame(frame)

        # Compute the violence probability regardless of label
        violence_pct = confidence if label == "violence" else 1.0 - confidence

        # Track consecutive violence detections above threshold
        if label == "violence" and confidence >= settings.confidence_threshold:
            consecutive_violence += 1
        else:
            consecutive_violence = 0

        # Determine the smoothed label for display and alerting
        is_confirmed_violence = consecutive_violence >= required_hits
        smoothed_label = "violence" if is_confirmed_violence else "normal"

        logger.debug(
            "Detection: %s (violence=%.1f%%) streak=%d/%d -> %s",
            label, violence_pct * 100, consecutive_violence, required_hits, smoothed_label,
        )

        # Update dashboard status
        app_state = getattr(detection_loop, '_app_state', None)
        if app_state:
            app_state.detector_status = {
                "label": smoothed_label,
                "confidence": confidence,
                "violence_score": round(violence_pct, 4),
                "streak": consecutive_violence,
                "required": required_hits,
                "last_update": time.time(),
            }

        # Signal clip recorder with the detection result
        clip_recorder.on_detection(is_confirmed_violence, confidence)

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
    clip_recorder = ClipRecorder(
        settings=settings,
        alert_manager=alert_manager,
        fps=camera.fps if camera.fps > 0 else 30.0,
    )

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
        args=(camera, clip_recorder, logger, stop_event),
        daemon=True,
    )
    capture_thread.start()
    logger.info("Capture thread started")

    # Start detection loop - runs inference periodically
    detection_thread = threading.Thread(
        target=detection_loop,
        args=(camera, detector, clip_recorder, settings, logger, stop_event),
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
