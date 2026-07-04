"""People and vehicle detection, tracking, and counting (YOLO + ByteTrack)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass(frozen=True)
class TrackedObject:
    """One tracked person or vehicle in the current frame."""

    track_id: int
    category: str  # "person" | "vehicle"
    label: str  # raw COCO class name, e.g. "car"
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class ObjectTracker:
    """Per-camera YOLO detector with ByteTrack multi-object tracking.

    Each camera pipeline owns its own instance: the tracker's ID state
    is tied to one video stream. Track IDs are persistent across frames,
    which powers unique-visitor counting and the behavior analytics.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
    ) -> None:
        # Imported lazily: ultralytics pulls in torch at import time
        from ultralytics import YOLO

        logger.info("Loading YOLO model %s...", model_path)
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._unique_people: set[int] = set()
        self._unique_vehicles: set[int] = set()
        self._last_visible = {"people": 0, "vehicles": 0}

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        """Run detection + tracking on one BGR frame.

        Returns the people and vehicles visible in this frame with
        persistent track IDs. Objects that ByteTrack has not confirmed
        yet (no ID assigned) are skipped.
        """
        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self._confidence,
            tracker="bytetrack.yaml",
        )
        result = results[0]
        boxes = result.boxes

        tracked: list[TrackedObject] = []
        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                label = result.names[int(boxes.cls[i])]
                if label in PERSON_CLASSES:
                    category = "person"
                elif label in VEHICLE_CLASSES:
                    category = "vehicle"
                else:
                    continue

                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
                tracked.append(
                    TrackedObject(
                        track_id=track_id,
                        category=category,
                        label=label,
                        confidence=float(boxes.conf[i]),
                        box=(x1, y1, x2, y2),
                    )
                )
                if category == "person":
                    self._unique_people.add(track_id)
                else:
                    self._unique_vehicles.add(track_id)

        self._last_visible = {
            "people": sum(1 for t in tracked if t.category == "person"),
            "vehicles": sum(1 for t in tracked if t.category == "vehicle"),
        }
        return tracked

    @property
    def counts(self) -> dict:
        """Current visible counts plus cumulative unique track counts."""
        return {
            "people": self._last_visible["people"],
            "vehicles": self._last_visible["vehicles"],
            "unique_people": len(self._unique_people),
            "unique_vehicles": len(self._unique_vehicles),
        }
