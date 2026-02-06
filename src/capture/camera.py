"""Camera capture with sliding window frame buffer."""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Reads frames from a webcam or video file and maintains a sliding window buffer.

    Parameters
    ----------
    source:
        Webcam index (int) or path to a video file (str).
    clip_length:
        Number of frames in the sliding window buffer.
    """

    def __init__(self, source: int | str = 0, clip_length: int = 16) -> None:
        self._source = source
        self._clip_length = clip_length
        self._buffer: deque[np.ndarray] = deque(maxlen=clip_length)
        self._cap = cv2.VideoCapture(source)
        self._released = False

        if not self._cap.isOpened():
            logger.warning("Failed to open video source: %s", source)

    # -- Frame I/O -----------------------------------------------------------

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the capture device.

        Returns
        -------
        (success, frame) where *success* is False when no frame is available.
        """
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def add_frame(self, frame: np.ndarray) -> None:
        """Push a frame into the sliding window buffer."""
        self._buffer.append(frame)

    def get_clip(self) -> Optional[np.ndarray]:
        """Return the current buffer as a numpy array of shape (clip_length, H, W, C).

        Returns None if the buffer is not yet full.
        """
        if len(self._buffer) < self._clip_length:
            return None
        return np.stack(list(self._buffer))

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently buffered frame, or None if buffer is empty."""
        if not self._buffer:
            return None
        return self._buffer[-1]

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
