"""Tests for the abnormal-behavior analytics."""

from __future__ import annotations

import math

import pytest

from src.analytics.behavior import BehaviorAnalyzer, FEATURE_COLUMNS
from src.detector.objects import TrackedObject


FRAME_SIZE = (480, 640)  # (height, width) -> diagonal = 800
DIAGONAL = math.hypot(640, 480)


def _person(track_id: int, x: float, y: float) -> TrackedObject:
    """A 20x40 person box centered at (x, y) in pixels."""
    return TrackedObject(
        track_id=track_id, category="person", label="person",
        confidence=0.9, box=(x - 10, y - 20, x + 10, y + 20),
    )


def _vehicle(track_id: int, x: float, y: float) -> TrackedObject:
    return TrackedObject(
        track_id=track_id, category="vehicle", label="car",
        confidence=0.9, box=(x - 30, y - 15, x + 30, y + 15),
    )


def _feed_track(analyzer, obj_factory, track_id, positions, t0=0.0, dt=0.2):
    """Feed a sequence of (x, y) positions for one track; return all events."""
    events = []
    for i, (x, y) in enumerate(positions):
        events.extend(
            analyzer.update([obj_factory(track_id, x, y)], FRAME_SIZE, now=t0 + i * dt)
        )
    return events


class TestRunningDetection:
    def test_fast_person_triggers_running(self):
        analyzer = BehaviorAnalyzer(run_speed=0.35)
        # 80 px per 0.2s step = 400 px/s = 0.5 diagonals/s > 0.35
        positions = [(50 + i * 80, 240) for i in range(6)]
        events = _feed_track(analyzer, _person, 1, positions)

        running = [e for e in events if e["type"] == "running"]
        assert running
        assert running[0]["track_id"] == 1
        assert running[0]["category"] == "person"
        assert 0 < running[0]["score"] <= 1

    def test_walking_person_does_not_trigger(self):
        analyzer = BehaviorAnalyzer(run_speed=0.35)
        # 10 px per 0.2s step = 50 px/s ~ 0.06 diagonals/s
        positions = [(50 + i * 10, 240) for i in range(6)]
        events = _feed_track(analyzer, _person, 1, positions)
        assert [e for e in events if e["type"] == "running"] == []

    def test_fast_vehicle_is_not_running(self):
        """Rule events only apply to people — vehicles are naturally fast."""
        analyzer = BehaviorAnalyzer(run_speed=0.35)
        positions = [(50 + i * 100, 240) for i in range(6)]
        events = _feed_track(analyzer, _vehicle, 1, positions)
        assert events == []

    def test_event_cooldown_limits_repeats(self):
        analyzer = BehaviorAnalyzer(run_speed=0.35, event_cooldown=30.0)
        positions = [(50 + i * 80, 240) for i in range(20)]
        events = _feed_track(analyzer, _person, 1, positions)
        assert len([e for e in events if e["type"] == "running"]) == 1


class TestLoiteringDetection:
    def test_stationary_person_triggers_loitering(self):
        analyzer = BehaviorAnalyzer(loiter_seconds=2.0, loiter_radius=0.08)
        # Person barely moves for 3 seconds (dt=0.2 -> 16 updates)
        positions = [(300 + (i % 2) * 3, 200) for i in range(16)]
        events = _feed_track(analyzer, _person, 5, positions)

        loitering = [e for e in events if e["type"] == "loitering"]
        assert loitering
        assert loitering[0]["track_id"] == 5

    def test_moving_person_does_not_loiter(self):
        analyzer = BehaviorAnalyzer(loiter_seconds=2.0, loiter_radius=0.08)
        # Wanders across the frame: radius far above the loiter threshold
        positions = [(50 + i * 40, 200) for i in range(16)]
        events = _feed_track(analyzer, _person, 5, positions)
        assert [e for e in events if e["type"] == "loitering"] == []

    def test_short_presence_does_not_loiter(self):
        analyzer = BehaviorAnalyzer(loiter_seconds=60.0)
        positions = [(300, 200)] * 6  # only ~1s of presence
        events = _feed_track(analyzer, _person, 5, positions)
        assert [e for e in events if e["type"] == "loitering"] == []


class TestAnomalyDetection:
    def test_model_fits_after_min_samples(self):
        analyzer = BehaviorAnalyzer(min_samples=20, refit_interval=20)
        assert analyzer.model_ready["person"] is False

        # Many normal walkers -> history fills -> model fits
        t = 0.0
        for track_id in range(10):
            for i in range(6):
                analyzer.update(
                    [_person(track_id, 50 + i * 10, 200)], FRAME_SIZE, now=t
                )
                t += 0.2
            t += 20.0  # let the old track expire between walkers

        assert analyzer.model_ready["person"] is True

    def test_erratic_track_flagged_as_anomalous(self):
        analyzer = BehaviorAnalyzer(
            min_samples=20, refit_interval=20, run_speed=99.0  # disable running rule
        )
        # Train on consistent slow walkers
        t = 0.0
        for track_id in range(12):
            for i in range(6):
                analyzer.update(
                    [_person(track_id, 50 + i * 10, 200)], FRAME_SIZE, now=t
                )
                t += 0.2
            t += 20.0
        assert analyzer.model_ready["person"] is True

        # A wildly erratic zig-zag sprinter
        erratic = [
            (100, 100), (600, 400), (80, 350), (620, 120), (60, 450), (600, 60),
        ]
        events = _feed_track(analyzer, _person, 999, erratic, t0=t)

        anomalies = [e for e in events if e["type"] == "anomalous_movement"]
        assert anomalies
        assert anomalies[0]["score"] > 0.6


class TestTrackPruning:
    def test_stale_tracks_removed(self):
        analyzer = BehaviorAnalyzer(track_ttl=10.0)
        analyzer.update([_person(1, 100, 100)], FRAME_SIZE, now=0.0)
        assert ("person", 1) in analyzer._tracks

        # A later frame without that track, past the TTL
        analyzer.update([], FRAME_SIZE, now=11.0)
        assert ("person", 1) not in analyzer._tracks


class TestFeatures:
    def test_features_require_three_points(self):
        analyzer = BehaviorAnalyzer()
        events = _feed_track(analyzer, _person, 1, [(100, 100), (110, 100)])
        # Not enough history for any feature computation -> no events, no history
        assert events == []
        assert analyzer._history["person"] == []

    def test_history_records_feature_columns(self):
        analyzer = BehaviorAnalyzer()
        _feed_track(analyzer, _person, 1, [(100 + i * 10, 100) for i in range(5)])
        assert analyzer._history["person"]
        assert len(analyzer._history["person"][0]) == len(FEATURE_COLUMNS)
