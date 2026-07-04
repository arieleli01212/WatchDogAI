"""ClipRecorder: state machine that captures short MP4 clips around detection events."""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.alerts.manager import AlertManager
from src.config import Settings

logger = logging.getLogger(__name__)


class _State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    SAVING = "saving"


class ClipRecorder:
    """Sits between the capture loop and alert manager for one camera.

    Keeps a rolling pre-event buffer and records full clips when an
    event is confirmed.  Thread-safe: ``add_frame`` is called from the
    capture thread and ``on_detection`` from the analysis thread.
    """

    def __init__(
        self,
        settings: Settings,
        alert_manager: AlertManager,
        fps: float = 30.0,
        camera_id: str = "cam0",
    ) -> None:
        self._settings = settings
        self._alert_manager = alert_manager
        self._fps = max(fps, 1.0)
        self._camera_id = camera_id

        pre_frames = int(self._fps * settings.pre_event_seconds)
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=max(pre_frames, 1))
        self._rec_frames: list[np.ndarray] = []

        self._clip_dir = Path(settings.clip_dir)
        self._clip_dir.mkdir(parents=True, exist_ok=True)

        self._state = _State.IDLE
        self._lock = threading.Lock()

        self._post_deadline: float | None = None
        self._rec_confidence: float = 0.0
        self._rec_alert_type: str = "violence"

    # ------------------------------------------------------------------
    # Public API (called from different threads)
    # ------------------------------------------------------------------

    def add_frame(self, frame: np.ndarray) -> None:
        """Feed a frame at full capture FPS."""
        with self._lock:
            if self._state is _State.IDLE:
                self._pre_buffer.append(frame)
            elif self._state is _State.RECORDING:
                self._rec_frames.append(frame)
                if self._post_deadline is not None and time.monotonic() >= self._post_deadline:
                    self._transition_to_saving()

    def on_detection(
        self,
        is_event: bool,
        confidence: float,
        alert_type: str = "violence",
    ) -> None:
        """Signal the current detection result from the analysis loop."""
        with self._lock:
            if self._state is _State.IDLE and is_event:
                self._start_recording(confidence, alert_type)
            elif self._state is _State.RECORDING:
                if is_event:
                    # Event continues — keep recording, reset post-deadline
                    self._post_deadline = None
                    self._rec_confidence = max(self._rec_confidence, confidence)
                else:
                    # Event ended — start post-event countdown
                    if self._post_deadline is None:
                        self._post_deadline = (
                            time.monotonic() + self._settings.post_event_seconds
                        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_recording(self, confidence: float, alert_type: str) -> None:
        """Transition from IDLE → RECORDING.  Must be called under lock."""
        logger.info(
            "ClipRecorder[%s]: %s detected, starting recording",
            self._camera_id, alert_type,
        )
        self._state = _State.RECORDING
        self._rec_confidence = confidence
        self._rec_alert_type = alert_type
        self._post_deadline = None
        # Copy pre-event buffer into recording frames
        self._rec_frames = list(self._pre_buffer)
        self._pre_buffer.clear()

    def _transition_to_saving(self) -> None:
        """Transition from RECORDING → SAVING.  Must be called under lock."""
        self._state = _State.SAVING
        frames = self._rec_frames
        confidence = self._rec_confidence
        alert_type = self._rec_alert_type

        # Reset recording state before releasing lock
        self._rec_frames = []
        self._post_deadline = None

        # Spawn a thread to write the file so we don't block the capture loop
        threading.Thread(
            target=self._write_clip,
            args=(frames, confidence, alert_type),
            daemon=True,
        ).start()

    def _write_clip(
        self,
        frames: list[np.ndarray],
        confidence: float,
        alert_type: str,
    ) -> None:
        """Write frames to MP4 and create an alert.  Runs in its own thread."""
        if not frames:
            with self._lock:
                self._state = _State.IDLE
            return

        now = datetime.now(tz=timezone.utc)
        date_dir = self._clip_dir / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{now.strftime('%H-%M-%S')}_{self._camera_id}.mp4"
        clip_path = date_dir / filename

        try:
            if not self._encode_clip(frames, clip_path):
                logger.error(
                    "ClipRecorder[%s]: failed to write clip %s — alert skipped",
                    self._camera_id, clip_path,
                )
                return

            logger.info(
                "ClipRecorder[%s]: saved %d frames to %s",
                self._camera_id, len(frames), clip_path,
            )

            self._alert_manager.on_clip_saved(
                confidence=confidence,
                clip_path=str(clip_path),
                camera_id=self._camera_id,
                alert_type=alert_type,
            )
        finally:
            with self._lock:
                self._state = _State.IDLE

    def _encode_clip(self, frames: list[np.ndarray], clip_path: Path) -> bool:
        """Encode frames to MP4. Returns False when the writer produced nothing usable."""
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return False
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return clip_path.is_file() and clip_path.stat().st_size > 0
