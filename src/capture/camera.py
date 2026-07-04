"""Camera capture: webcam, video file, or network stream with reconnect logic."""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://")


class SourceType(enum.Enum):
    WEBCAM = "webcam"
    FILE = "file"
    STREAM = "stream"


def classify_source(source: int | str) -> SourceType:
    """Classify a capture source: webcam index, network stream URL, or file path."""
    if isinstance(source, int):
        return SourceType.WEBCAM
    if source.lower().startswith(STREAM_PREFIXES):
        return SourceType.STREAM
    return SourceType.FILE


class Camera:
    """Reads frames from a webcam, video file, or network stream.

    Loops video files, reconnects dropped streams with exponential
    backoff, applies the configured capture quality, and keeps the
    latest frame with a sequence counter so consumers can tell new
    frames from already-seen ones.

    Parameters
    ----------
    source:
        Webcam index (int), video file path, or stream URL.
    camera_id / name:
        Identity attached to alerts, clips, and dashboard routes.
    width / height / target_fps:
        Requested capture quality; 0 keeps the source default.
    """

    # After this many consecutive failed reads a webcam/stream source is reopened.
    FAILURES_BEFORE_RECONNECT = 5
    RECONNECT_BASE_DELAY = 0.5
    RECONNECT_MAX_DELAY = 30.0

    def __init__(
        self,
        source: int | str = 0,
        camera_id: str = "cam0",
        name: str = "",
        width: int = 0,
        height: int = 0,
        target_fps: float = 0.0,
    ) -> None:
        self.id = camera_id
        self.name = name or camera_id
        self._source = source
        self._source_type = classify_source(source)
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._released = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_frame_time: Optional[float] = None
        self._consecutive_failures = 0

        self._cap = self._open()

    # -- Source lifecycle ------------------------------------------------------

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source)
        if cap.isOpened():
            self._configure(cap)
            logger.info(
                "Camera %s: opened %s source %s",
                self.id, self._source_type.value, self._source,
            )
        else:
            logger.warning(
                "Camera %s: failed to open video source: %s", self.id, self._source
            )
        return cap

    def _configure(self, cap: cv2.VideoCapture) -> None:
        if self._width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._target_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        if self._source_type is SourceType.STREAM:
            # Keep only the freshest frame; stale buffered frames add latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # -- Frame I/O -----------------------------------------------------------

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single raw frame from the capture device.

        Returns
        -------
        (success, frame) where *success* is False when no frame is available.
        """
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame, transparently handling source recovery.

        Video files loop back to the start when they end; webcams and
        network streams are reopened with exponential backoff after
        repeated failures. Returns None when no frame is available this
        iteration (the caller should simply retry). May sleep briefly, so
        call this from a dedicated capture thread.
        """
        if self._released:
            return None

        ret, frame = self._cap.read()
        if ret:
            self._consecutive_failures = 0
            if self._source_type is SourceType.FILE and self.fps > 0:
                # Pace file playback to its native FPS
                time.sleep(1.0 / self.fps)
            return frame

        self._consecutive_failures += 1
        self._recover()
        return None

    def _recover(self) -> None:
        """Handle a failed read according to the source type."""
        if self._source_type is SourceType.FILE and self._cap.isOpened():
            # End of file: loop back to the beginning
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.debug("Camera %s: video ended, looping", self.id)
            return

        if self._consecutive_failures < self.FAILURES_BEFORE_RECONNECT:
            time.sleep(0.05)
            return

        attempt = self._consecutive_failures - self.FAILURES_BEFORE_RECONNECT
        delay = min(
            self.RECONNECT_BASE_DELAY * (2 ** min(attempt, 6)),
            self.RECONNECT_MAX_DELAY,
        )
        logger.warning(
            "Camera %s: %d consecutive read failures, reopening source in %.1fs",
            self.id, self._consecutive_failures, delay,
        )
        time.sleep(delay)
        self._cap.release()
        self._cap = self._open()

    def add_frame(self, frame: np.ndarray) -> None:
        """Publish a captured frame as the latest frame."""
        with self._lock:
            self._latest_frame = frame
            self._seq += 1
            self._last_frame_time = time.monotonic()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame, or None if no frames yet."""
        with self._lock:
            return self._latest_frame

    def get_latest_frame_with_seq(self) -> tuple[Optional[np.ndarray], int]:
        """Return (latest frame, sequence number).

        The sequence number increments once per captured frame, letting
        consumers skip frames they have already processed.
        """
        with self._lock:
            return self._latest_frame, self._seq

    # -- Health ----------------------------------------------------------------

    @property
    def last_frame_age(self) -> Optional[float]:
        """Seconds since the last captured frame, or None if none yet."""
        with self._lock:
            if self._last_frame_time is None:
                return None
            return time.monotonic() - self._last_frame_time

    def is_healthy(self, max_age_seconds: float = 5.0) -> bool:
        """Return True if the camera produced a frame recently."""
        age = self.last_frame_age
        return age is not None and age <= max_age_seconds

    # -- Lifecycle ------------------------------------------------------------

    def is_opened(self) -> bool:
        """Return True if the capture device is open and not yet released."""
        if self._released:
            return False
        return self._cap.isOpened()

    def release(self) -> None:
        """Release the underlying capture device."""
        self._cap.release()
        self._released = True

    # -- Properties -----------------------------------------------------------

    @property
    def source_type(self) -> SourceType:
        """The classified source type (webcam, file, or stream)."""
        return self._source_type

    @property
    def fps(self) -> float:
        """Frames per second reported by the capture device."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_width(self) -> int:
        """Frame width in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Frame height in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -- Context manager ------------------------------------------------------

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.release()
