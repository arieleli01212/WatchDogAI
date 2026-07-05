"""Per-camera processing pipeline: capture and analysis threads."""

from __future__ import annotations

import logging
import threading
import time

from src.alerts.clip_recorder import ClipRecorder
from src.alerts.manager import AlertManager
from src.analytics.behavior import BehaviorAnalyzer
from src.capture.camera import Camera
from src.config import CameraConfig, Settings
from src.detector.model import ViolenceDetector
from src.detector.objects import ObjectTracker

logger = logging.getLogger(__name__)


class CameraPipeline:
    """Owns one camera and its worker threads.

    The *capture thread* reads frames at full source speed into the
    camera buffer and the clip recorder. The *analysis thread* classifies
    each new frame (skipping frames it has already seen via the camera's
    sequence counter), applies temporal smoothing, publishes per-camera
    status into ``status_registry``, and signals the clip recorder.

    Heavy models (the violence detector) are shared across pipelines;
    per-camera state (camera, clip recorder, smoothing streak) is owned
    here.
    """

    def __init__(
        self,
        config: CameraConfig,
        settings: Settings,
        detector: ViolenceDetector,
        alert_manager: AlertManager,
        status_registry: dict,
        stop_event: threading.Event,
    ) -> None:
        self.config = config
        self._settings = settings
        self._detector = detector
        self._status = status_registry
        self._stop = stop_event

        self.camera = Camera(
            source=config.source,
            camera_id=config.id,
            name=config.name,
            width=config.width,
            height=config.height,
            target_fps=config.fps,
        )
        self.clip_recorder = ClipRecorder(
            settings=settings,
            alert_manager=alert_manager,
            fps=self.camera.fps if self.camera.fps > 0 else 30.0,
            camera_id=config.id,
        )
        # Tracker state (track IDs) is per-stream, so each pipeline owns one
        self._tracker: ObjectTracker | None = None
        if settings.object_detection_enabled:
            try:
                self._tracker = ObjectTracker(
                    model_path=settings.yolo_model,
                    confidence=settings.yolo_confidence,
                )
            except Exception:
                logger.exception(
                    "Camera %s: object tracker unavailable, running without "
                    "people/vehicle analytics", config.id,
                )

        # Behavior analytics need trajectories, so they require the tracker
        self._behavior: BehaviorAnalyzer | None = None
        if settings.behavior_enabled and self._tracker is not None:
            self._behavior = BehaviorAnalyzer(
                loiter_seconds=settings.loiter_seconds,
                run_speed=settings.run_speed_threshold,
                min_samples=settings.anomaly_min_samples,
                event_cooldown=settings.behavior_event_cooldown,
            )
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the capture and analysis threads."""
        for target, suffix in (
            (self._capture_loop, "capture"),
            (self._analysis_loop, "analysis"),
        ):
            thread = threading.Thread(
                target=target, name=f"{self.config.id}-{suffix}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def join(self, timeout: float = 5.0) -> None:
        """Wait for the worker threads to finish."""
        for thread in self._threads:
            thread.join(timeout)

    def release(self) -> None:
        """Release the underlying camera."""
        self.camera.release()

    # ------------------------------------------------------------------
    # Worker loops
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        logger.info("Camera %s: capture loop started", self.config.id)
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                continue
            self.camera.add_frame(frame)
            self.clip_recorder.add_frame(frame)
        logger.info("Camera %s: capture loop stopped", self.config.id)

    def _analysis_loop(self) -> None:
        """Classify new frames with temporal smoothing.

        Only triggers an alert after N consecutive high-confidence
        violence detections on *distinct* frames — the sequence counter
        guarantees the same frame is never counted twice.
        """
        logger.info("Camera %s: analysis loop started", self.config.id)
        consecutive_violence = 0
        required_hits = self._settings.consecutive_hits
        last_seq = 0

        while not self._stop.is_set():
            frame, seq = self.camera.get_latest_frame_with_seq()
            if frame is None or seq == last_seq:
                # Even without new frames, let an overdue recording finish
                # (e.g. the camera died right after the incident)
                self.clip_recorder.tick()
                time.sleep(0.01)
                continue
            last_seq = seq

            label, confidence = self._detector.predict_frame(frame)
            violence_score = confidence if label == "violence" else 1.0 - confidence

            if label == "violence" and confidence >= self._settings.confidence_threshold:
                consecutive_violence += 1
            else:
                consecutive_violence = 0

            is_confirmed = consecutive_violence >= required_hits

            tracked = self._track_objects(frame)
            counts = self._tracker.counts if self._tracker else {}
            behavior_events = self._analyze_behavior(tracked, frame.shape[:2])

            logger.debug(
                "Camera %s: %s (violence=%.1f%%) streak=%d/%d people=%d vehicles=%d",
                self.config.id, label, violence_score * 100,
                consecutive_violence, required_hits,
                counts.get("people", 0), counts.get("vehicles", 0),
            )

            self._status[self.config.id] = {
                "label": "violence" if is_confirmed else "normal",
                "confidence": confidence,
                "violence_score": round(violence_score, 4),
                "streak": consecutive_violence,
                "required": required_hits,
                "counts": counts,
                "objects": [
                    {
                        "track_id": t.track_id,
                        "category": t.category,
                        "label": t.label,
                        "confidence": round(t.confidence, 3),
                        "box": [round(v, 1) for v in t.box],
                    }
                    for t in tracked
                ],
                "behavior_events": behavior_events,
                "last_update": time.time(),
            }

            # Violence takes precedence; otherwise behavior events trigger clips
            if is_confirmed:
                self.clip_recorder.on_detection(True, confidence, alert_type="violence")
            elif behavior_events:
                top = max(behavior_events, key=lambda e: e["score"])
                logger.info(
                    "Camera %s: behavior event %s (track %s): %s",
                    self.config.id, top["type"], top["track_id"], top["details"],
                )
                self.clip_recorder.on_detection(
                    True, top["score"], alert_type=top["type"]
                )
            else:
                self.clip_recorder.on_detection(False, confidence)

        logger.info("Camera %s: analysis loop stopped", self.config.id)

    def _track_objects(self, frame) -> list:
        """Run people/vehicle tracking; returns TrackedObjects for this frame."""
        if self._tracker is None:
            return []
        try:
            return self._tracker.update(frame)
        except Exception:
            logger.exception(
                "Camera %s: object tracking failed, disabling it", self.config.id
            )
            self._tracker = None
            return []

    def _analyze_behavior(self, tracked: list, frame_size: tuple[int, int]) -> list[dict]:
        """Run behavior analytics on the tracked objects for this frame."""
        if self._behavior is None:
            return []
        try:
            return self._behavior.update(tracked, frame_size)
        except Exception:
            logger.exception(
                "Camera %s: behavior analytics failed, disabling them", self.config.id
            )
            self._behavior = None
            return []
