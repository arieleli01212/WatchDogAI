"""Runtime pipeline management: build, stop, and swap camera pipelines.

The dashboard's source-mode toggle (live cameras vs. a folder of
recordings) needs the set of running pipelines to change without
restarting the process. PipelineManager owns that lifecycle: it holds
the current generation of CameraPipelines plus their shared stop event,
and swaps generations atomically under a lock.

The camera and status registries are the dicts the dashboard routes and
the MQTT telemetry loop already hold references to, so they are always
mutated in place — never rebound — to keep every reader current.
"""

from __future__ import annotations

import logging
import threading

from src.alerts.manager import AlertManager
from src.config import CameraConfig, Settings, recordings_camera_configs
from src.detector.model import ViolenceDetector
from src.pipeline import CameraPipeline

logger = logging.getLogger(__name__)

MODE_LIVE = "live"
MODE_RECORDINGS = "recordings"


class PipelineManager:
    """Owns the running camera pipelines and switches their source mode.

    - ``live`` — the cameras configured via CAMERAS / CAMERA_SOURCE.
    - ``recordings`` — one camera per video file in RECORDINGS_DIR
      (only offered when that setting points at a folder; the folder is
      re-scanned on every switch, so newly added files are picked up).
    """

    def __init__(
        self,
        settings: Settings,
        detector: ViolenceDetector,
        alert_manager: AlertManager,
        cameras_registry: dict,
        status_registry: dict,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._alert_manager = alert_manager
        self._cameras = cameras_registry
        self._status = status_registry

        self._pipelines: list[CameraPipeline] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._mode = MODE_LIVE

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def available_modes(self) -> list[str]:
        modes = [MODE_LIVE]
        if self._settings.recordings_dir:
            modes.append(MODE_RECORDINGS)
        return modes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, mode: str) -> None:
        """(Re)build all pipelines for *mode*, tearing down any running ones.

        Opening cameras is synchronous and can take seconds for
        unreachable network sources — call from a worker thread, not the
        event loop.
        """
        if mode not in self.available_modes():
            raise ValueError(f"unknown source mode {mode!r}")

        configs = self._configs_for(mode)

        with self._lock:
            self._teardown_locked()
            self._stop_event = threading.Event()
            for config in configs:
                pipeline = CameraPipeline(
                    config=config,
                    settings=self._settings,
                    detector=self._detector,
                    alert_manager=self._alert_manager,
                    status_registry=self._status,
                    stop_event=self._stop_event,
                )
                self._cameras[config.id] = pipeline.camera
                self._pipelines.append(pipeline)
            for pipeline in self._pipelines:
                pipeline.start()
            self._mode = mode
            logger.info(
                "Source mode %r active with %d camera(s): %s",
                mode, len(self._pipelines),
                ", ".join(p.config.id for p in self._pipelines),
            )

    def switch(self, mode: str) -> bool:
        """Switch to *mode*; returns False when it is already active."""
        if mode not in self.available_modes():
            raise ValueError(f"unknown source mode {mode!r}")
        if mode == self._mode and self._pipelines:
            return False
        self.start(mode)
        return True

    def stop(self) -> None:
        """Tear down all running pipelines (used at shutdown)."""
        with self._lock:
            self._teardown_locked()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _teardown_locked(self) -> None:
        """Stop and release the current generation. Must hold the lock."""
        if not self._pipelines:
            return
        logger.info(
            "Stopping %d pipeline(s) for source mode %r",
            len(self._pipelines), self._mode,
        )
        self._stop_event.set()
        for pipeline in self._pipelines:
            pipeline.join(timeout=5)
        for pipeline in self._pipelines:
            pipeline.release()
        self._pipelines = []
        # Mutate in place: the dashboard and telemetry loop hold references
        self._cameras.clear()
        self._status.clear()

    def _configs_for(self, mode: str) -> tuple[CameraConfig, ...]:
        if mode == MODE_RECORDINGS:
            return recordings_camera_configs(self._settings.recordings_dir)
        return self._settings.cameras
