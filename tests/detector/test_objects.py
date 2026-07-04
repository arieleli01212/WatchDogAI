"""Tests for people/vehicle detection, tracking, and counting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector.objects import ObjectTracker, TrackedObject


FRAME = np.zeros((64, 64, 3), dtype=np.uint8)

COCO_NAMES = {0: "person", 2: "car", 5: "bus", 16: "dog"}


class FakeBoxes:
    def __init__(self, ids, cls, conf, xyxy):
        self.id = ids
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = COCO_NAMES


def _make_tracker(results_sequence):
    """Build an ObjectTracker whose YOLO model yields the given results."""
    with patch("ultralytics.YOLO") as mock_yolo_cls:
        model = mock_yolo_cls.return_value
        model.track.side_effect = [[r] for r in results_sequence]
        tracker = ObjectTracker(model_path="fake.pt", confidence=0.4)
    return tracker, model


class TestTrackedObject:
    def test_center(self):
        obj = TrackedObject(
            track_id=1, category="person", label="person",
            confidence=0.9, box=(10.0, 20.0, 30.0, 40.0),
        )
        assert obj.center == (20.0, 30.0)


class TestUpdate:
    def test_people_and_vehicles_tracked(self):
        boxes = FakeBoxes(
            ids=[1, 2, 3],
            cls=[0, 2, 16],  # person, car, dog
            conf=[0.9, 0.8, 0.7],
            xyxy=[[0, 0, 10, 10], [10, 10, 30, 30], [40, 40, 50, 50]],
        )
        tracker, model = _make_tracker([FakeResult(boxes)])

        tracked = tracker.update(FRAME)

        # The dog is not a person or vehicle and is dropped
        assert len(tracked) == 2
        person, car = tracked
        assert person.category == "person"
        assert person.track_id == 1
        assert car.category == "vehicle"
        assert car.label == "car"
        assert car.box == (10.0, 10.0, 30.0, 30.0)

    def test_track_called_with_bytetrack_and_persist(self):
        tracker, model = _make_tracker([FakeResult(FakeBoxes([], [], [], []))])
        tracker.update(FRAME)

        kwargs = model.track.call_args.kwargs
        assert kwargs["persist"] is True
        assert kwargs["tracker"] == "bytetrack.yaml"
        assert kwargs["conf"] == 0.4

    def test_unconfirmed_tracks_skipped(self):
        """Boxes without ByteTrack IDs (id=None) are ignored."""
        boxes = FakeBoxes(ids=None, cls=[0], conf=[0.9], xyxy=[[0, 0, 5, 5]])
        tracker, _ = _make_tracker([FakeResult(boxes)])
        assert tracker.update(FRAME) == []

    def test_no_boxes(self):
        result = FakeResult(None)
        tracker, _ = _make_tracker([result])
        assert tracker.update(FRAME) == []


class TestCounts:
    def test_visible_counts(self):
        boxes = FakeBoxes(
            ids=[1, 2, 3],
            cls=[0, 0, 5],  # two people, one bus
            conf=[0.9, 0.9, 0.8],
            xyxy=[[0, 0, 1, 1]] * 3,
        )
        tracker, _ = _make_tracker([FakeResult(boxes)])
        tracker.update(FRAME)

        counts = tracker.counts
        assert counts["people"] == 2
        assert counts["vehicles"] == 1

    def test_unique_counts_accumulate_across_frames(self):
        frame1 = FakeResult(FakeBoxes([1, 2], [0, 0], [0.9, 0.9], [[0, 0, 1, 1]] * 2))
        frame2 = FakeResult(FakeBoxes([2, 3], [0, 2], [0.9, 0.9], [[0, 0, 1, 1]] * 2))
        tracker, _ = _make_tracker([frame1, frame2])

        tracker.update(FRAME)
        tracker.update(FRAME)

        counts = tracker.counts
        # Track 2 seen twice but counted once; tracks 1, 2, 3 -> 2 people + 1 vehicle
        assert counts["unique_people"] == 2
        assert counts["unique_vehicles"] == 1
        # Visible counts reflect only the latest frame
        assert counts["people"] == 1
        assert counts["vehicles"] == 1

    def test_counts_before_first_update(self):
        tracker, _ = _make_tracker([])
        assert tracker.counts == {
            "people": 0, "vehicles": 0, "unique_people": 0, "unique_vehicles": 0,
        }
