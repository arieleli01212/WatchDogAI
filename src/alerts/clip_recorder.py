"""ClipRecorder: state machine that captures short MP4 clips around detection events."""

from __future__ import annotations

import enum
import logging
import re
import shutil
import subprocess
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

# Direct H.264 writer candidates, tried in order. Browsers only play
# H.264 ("avc1") in <video>, not MPEG-4 Part 2 ("mp4v") — on Windows the
# Media Foundation backend provides H.264 natively, no ffmpeg needed.
H264_WRITER_CANDIDATES: tuple[tuple[str, int | None], ...] = (
    ("avc1", getattr(cv2, "CAP_MSMF", None)),
    ("avc1", None),
)


class _State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    SAVING = "saving"


class ClipRecorder:
    """Sits between the capture loop and alert manager for one camera.

    Keeps a rolling pre-event buffer and records full clips when an
    event is confirmed. Thread-safe: ``add_frame`` is called from the
    capture thread, ``on_detection``/``tick`` from the analysis thread,
    and clips are encoded on short-lived writer threads.

    Robustness properties:

    - Recording length is capped (``max_clip_seconds``) so a sustained
      event chunks into multiple clips instead of growing in memory.
    - ``tick()`` finalizes an overdue recording even when the camera
      stops delivering frames (e.g. it was disabled right after the
      incident), so captured footage is never stranded in RECORDING.
    - While a clip is being written (SAVING) the pre-event buffer keeps
      filling and a newly confirmed event starts a fresh recording
      immediately — back-to-back events are not lost.
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
        # camera_id appears in filenames; strip anything path-hostile
        self._safe_camera_id = re.sub(r"[^A-Za-z0-9_-]+", "_", camera_id) or "camera"

        pre_frames = int(self._fps * settings.pre_event_seconds)
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=max(pre_frames, 1))
        self._rec_frames: list[np.ndarray] = []
        self._max_rec_frames = max(int(self._fps * settings.max_clip_seconds), 1)

        self._clip_dir = Path(settings.clip_dir)
        self._clip_dir.mkdir(parents=True, exist_ok=True)

        self._state = _State.IDLE
        self._lock = threading.Lock()

        self._post_deadline: float | None = None
        self._rec_confidence: float = 0.0
        self._rec_alert_type: str = "violence"
        # Direct-H.264 probe result, cached after the first clip:
        # None = not probed yet, False = unavailable on this system,
        # (fourcc, api) = the working writer combination
        self._direct_h264: tuple[str, int | None] | bool | None = None

    # ------------------------------------------------------------------
    # Public API (called from different threads)
    # ------------------------------------------------------------------

    def add_frame(self, frame: np.ndarray) -> None:
        """Feed a frame at full capture FPS."""
        with self._lock:
            if self._state is _State.RECORDING:
                self._rec_frames.append(frame)
                self._check_recording_limits()
            else:
                # IDLE and SAVING both keep the pre-event context warm so
                # a follow-up event still gets its lead-in footage
                self._pre_buffer.append(frame)

    def on_detection(
        self,
        is_event: bool,
        confidence: float,
        alert_type: str = "violence",
    ) -> None:
        """Signal the current detection result from the analysis loop."""
        with self._lock:
            if self._state is _State.RECORDING:
                if is_event:
                    # Event continues — keep recording, reset post-deadline
                    self._post_deadline = None
                    self._rec_confidence = max(self._rec_confidence, confidence)
                    # Violence outranks behavior events in the recorded type
                    if alert_type == "violence":
                        self._rec_alert_type = "violence"
                else:
                    # Event ended — start post-event countdown
                    if self._post_deadline is None:
                        self._post_deadline = (
                            time.monotonic() + self._settings.post_event_seconds
                        )
                    self._check_recording_limits()
            elif is_event:
                # IDLE or SAVING: an in-flight write does not block a new
                # event (the writer thread owns its own frame snapshot)
                self._start_recording(confidence, alert_type)

    def tick(self) -> None:
        """Finalize an overdue recording even when no frames arrive."""
        with self._lock:
            if self._state is _State.RECORDING:
                self._check_recording_limits()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_recording_limits(self) -> None:
        """Transition to SAVING when overdue or too long. Must hold the lock."""
        if self._post_deadline is not None and time.monotonic() >= self._post_deadline:
            self._transition_to_saving()
            return
        if len(self._rec_frames) >= self._max_rec_frames:
            logger.warning(
                "ClipRecorder[%s]: max clip length (%ss) reached, chunking",
                self._camera_id, self._settings.max_clip_seconds,
            )
            self._transition_to_saving()

    def _start_recording(self, confidence: float, alert_type: str) -> None:
        """Transition to RECORDING.  Must be called under lock."""
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
        try:
            if not frames:
                return

            now = datetime.now(tz=timezone.utc)
            date_dir = self._clip_dir / now.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            # Millisecond suffix keeps chunked back-to-back clips distinct
            filename = (
                f"{now.strftime('%H-%M-%S')}-{now.microsecond // 1000:03d}"
                f"_{self._safe_camera_id}.mp4"
            )
            clip_path = date_dir / filename

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
                # A new event may have re-entered RECORDING while we wrote;
                # only clear the state if no one else claimed it
                if self._state is _State.SAVING:
                    self._state = _State.IDLE

    def _encode_clip(self, frames: list[np.ndarray], clip_path: Path) -> bool:
        """Encode frames to a browser-playable MP4.

        Browsers only decode H.264 in <video>; OpenCV's default ``mp4v``
        (MPEG-4 Part 2) renders as a dead black player on the dashboard.
        Three tiers, best first:

        1. Direct H.264 via an OpenCV writer (Windows Media Foundation
           provides this natively) — single pass, no external tools.
        2. ``mp4v`` to a temp file, re-encoded to H.264 by ffmpeg when
           it is installed (typical on Linux servers).
        3. Keep the ``mp4v`` clip — still plays in VLC and still uploads
           to the control center, with a warning about browser previews.

        Returns False only when nothing usable was written.
        """
        h, w = frames[0].shape[:2]

        # Tier 1: direct H.264 (probed once, then cached per recorder)
        if self._direct_h264 is not False:
            candidates = (
                [self._direct_h264]
                if self._direct_h264
                else list(H264_WRITER_CANDIDATES)
            )
            for fourcc_str, api in candidates:
                if self._write_with(clip_path, fourcc_str, api, frames, w, h):
                    self._direct_h264 = (fourcc_str, api)
                    return True
            self._direct_h264 = False
            logger.warning(
                "ClipRecorder[%s]: no native H.264 encoder available — "
                "falling back to mp4v (+ ffmpeg re-encode when installed)",
                self._camera_id,
            )

        # Tier 2/3: mp4v temp file, then ffmpeg re-encode or keep as-is
        tmp_path = clip_path.with_name(clip_path.stem + ".tmp.mp4")
        if not self._write_with(tmp_path, "mp4v", None, frames, w, h):
            tmp_path.unlink(missing_ok=True)
            return False

        if not self._reencode_h264(tmp_path, clip_path):
            tmp_path.replace(clip_path)  # fallback: keep the mp4v clip
        return clip_path.is_file() and clip_path.stat().st_size > 0

    def _write_with(
        self,
        path: Path,
        fourcc_str: str,
        api: int | None,
        frames: list[np.ndarray],
        w: int,
        h: int,
    ) -> bool:
        """Write frames with one specific fourcc/backend; validate the output."""
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        if api is None:
            writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        else:
            writer = cv2.VideoWriter(str(path), api, fourcc, self._fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return False
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        if path.is_file() and path.stat().st_size > 0:
            return True
        path.unlink(missing_ok=True)
        return False

    def _reencode_h264(self, src: Path, dst: Path) -> bool:
        """Re-encode *src* to browser-playable H.264 at *dst*. Cleans up *src* on success."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            logger.warning(
                "ClipRecorder[%s]: ffmpeg not found — clip %s kept as mp4v "
                "(dashboard previews won't play it)",
                self._camera_id, dst.name,
            )
            return False
        result = subprocess.run(
            [
                ffmpeg, "-y", "-v", "error",
                "-i", str(src),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(dst),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not (dst.is_file() and dst.stat().st_size > 0):
            logger.warning(
                "ClipRecorder[%s]: H.264 re-encode failed (%s) — keeping mp4v clip",
                self._camera_id, result.stderr.strip() or f"exit {result.returncode}",
            )
            dst.unlink(missing_ok=True)
            return False
        src.unlink(missing_ok=True)
        return True
