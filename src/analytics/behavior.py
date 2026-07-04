"""Suspicious-movement and abnormal-behavior analytics on tracked trajectories.

Consumes per-frame tracked objects (people/vehicles with persistent IDs)
and emits behavior events through two complementary detectors:

1. Rules — explainable, always active:
   - *loitering*: a person present longer than ``loiter_seconds`` while
     staying within a small radius of their average position.
   - *running*: a person whose smoothed speed exceeds
     ``run_speed`` (measured in frame diagonals per second, so the
     threshold is resolution-independent).

2. Statistics — an unsupervised scikit-learn IsolationForest fitted on
   the rolling history of motion features observed by this camera,
   combined with learned per-feature range bounds. Once enough samples
   are collected, feature vectors that the model isolates — or that fall
   far outside every speed/tortuosity value previously observed — are
   reported as *anomalous_movement*. The range guard matters because an
   isolation forest cannot split on features whose training values are
   nearly constant, so it under-scores out-of-range outliers when the
   camera has only seen uniform motion so far.

All coordinates are normalized by the frame diagonal so the same
configuration works across cameras with different resolutions.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detector.objects import TrackedObject

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["speed_mean", "speed_max", "tortuosity", "dwell_time", "radius"]

# Movement-style features checked against learned bounds. dwell_time and
# radius grow naturally over a track's lifetime, so they are excluded.
RANGE_GUARD_COLUMNS = ["speed_mean", "speed_max", "tortuosity"]

# Rule-event confidences reported in alerts
LOITERING_SCORE = 0.80
RUNNING_SCORE = 0.75


@dataclass
class _TrackState:
    """Motion history for one tracked object."""

    first_seen: float
    positions: deque = field(default_factory=lambda: deque(maxlen=600))  # (t, x, y)
    last_seen: float = 0.0
    last_event_times: dict = field(default_factory=dict)


class BehaviorAnalyzer:
    """Per-camera behavior analytics over tracked-object trajectories."""

    def __init__(
        self,
        loiter_seconds: float = 60.0,
        loiter_radius: float = 0.08,
        run_speed: float = 0.35,
        window_seconds: float = 3.0,
        min_samples: int = 200,
        refit_interval: int = 100,
        event_cooldown: float = 30.0,
        track_ttl: float = 10.0,
        max_history: int = 5000,
    ) -> None:
        self._loiter_seconds = loiter_seconds
        self._loiter_radius = loiter_radius
        self._run_speed = run_speed
        self._window_seconds = window_seconds
        self._min_samples = min_samples
        self._refit_interval = refit_interval
        self._event_cooldown = event_cooldown
        self._track_ttl = track_ttl
        self._max_history = max_history

        self._tracks: dict[tuple[str, int], _TrackState] = {}
        self._history: dict[str, list[list[float]]] = {"person": [], "vehicle": []}
        self._models: dict[str, IsolationForest | None] = {"person": None, "vehicle": None}
        self._bounds: dict[str, dict[str, tuple[float, float]]] = {"person": {}, "vehicle": {}}
        self._samples_since_fit: dict[str, int] = {"person": 0, "vehicle": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        objects: list[TrackedObject],
        frame_size: tuple[int, int],
        now: float | None = None,
    ) -> list[dict]:
        """Feed one frame's tracked objects; returns triggered behavior events.

        Parameters
        ----------
        objects:
            Tracked people/vehicles for the current frame.
        frame_size:
            (height, width) of the frame, used to normalize coordinates.
        now:
            Timestamp override for tests; defaults to ``time.monotonic()``.
        """
        now = time.monotonic() if now is None else now
        height, width = frame_size
        diagonal = math.hypot(width, height) or 1.0

        events: list[dict] = []
        for obj in objects:
            key = (obj.category, obj.track_id)
            state = self._tracks.get(key)
            if state is None:
                state = self._tracks[key] = _TrackState(first_seen=now)
            state.last_seen = now
            cx, cy = obj.center
            state.positions.append((now, cx / diagonal, cy / diagonal))

            features = self._compute_features(state, now)
            if features is None:
                continue

            events.extend(self._rule_events(obj, state, features, now))
            anomaly = self._anomaly_event(obj, state, features, now)
            if anomaly is not None:
                events.append(anomaly)

            self._record_sample(obj.category, features)

        self._prune_stale_tracks(now)
        return events

    @property
    def model_ready(self) -> dict[str, bool]:
        """Whether the anomaly model has been fitted, per category."""
        return {cat: model is not None for cat, model in self._models.items()}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _compute_features(self, state: _TrackState, now: float) -> dict | None:
        """Motion features over the recent window; None until enough points."""
        window = [
            (t, x, y)
            for t, x, y in state.positions
            if now - t <= self._window_seconds
        ]
        if len(window) < 3:
            return None

        speeds: list[float] = []
        path_length = 0.0
        for (t1, x1, y1), (t2, x2, y2) in zip(window, window[1:]):
            dist = math.hypot(x2 - x1, y2 - y1)
            path_length += dist
            dt = t2 - t1
            if dt > 0:
                speeds.append(dist / dt)
        if not speeds:
            return None

        displacement = math.hypot(
            window[-1][1] - window[0][1], window[-1][2] - window[0][2]
        )
        # Path/displacement ratio: ~1 for straight walking, high for erratic motion
        tortuosity = min(path_length / max(displacement, 1e-6), 20.0)

        # Lifetime radius: max distance from the track's average position
        xs = [x for _, x, _ in state.positions]
        ys = [y for _, _, y in state.positions]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        radius = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))

        return {
            "speed_mean": sum(speeds) / len(speeds),
            "speed_max": max(speeds),
            "tortuosity": tortuosity,
            "dwell_time": now - state.first_seen,
            "radius": radius,
        }

    # ------------------------------------------------------------------
    # Rule-based detectors
    # ------------------------------------------------------------------

    def _rule_events(
        self, obj: TrackedObject, state: _TrackState, features: dict, now: float
    ) -> list[dict]:
        events: list[dict] = []
        if obj.category != "person":
            return events

        if (
            features["dwell_time"] >= self._loiter_seconds
            and features["radius"] <= self._loiter_radius
            and self._cooldown_ok(state, "loitering", now)
        ):
            events.append(
                self._event(
                    "loitering", obj, LOITERING_SCORE,
                    f"present {features['dwell_time']:.0f}s within radius "
                    f"{features['radius']:.3f} of frame diagonal",
                )
            )

        if features["speed_mean"] >= self._run_speed and self._cooldown_ok(
            state, "running", now
        ):
            events.append(
                self._event(
                    "running", obj, RUNNING_SCORE,
                    f"speed {features['speed_mean']:.2f} diagonals/s "
                    f"(threshold {self._run_speed})",
                )
            )
        return events

    # ------------------------------------------------------------------
    # Statistical anomaly detector
    # ------------------------------------------------------------------

    def _anomaly_event(
        self, obj: TrackedObject, state: _TrackState, features: dict, now: float
    ) -> dict | None:
        model = self._models.get(obj.category)
        if model is None:
            return None

        bounds = self._bounds.get(obj.category, {})
        violations = [
            col
            for col in RANGE_GUARD_COLUMNS
            if col in bounds
            and not (bounds[col][0] <= features[col] <= bounds[col][1])
        ]

        row = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        score = float(model.decision_function(row)[0])
        if score >= 0 and not violations:  # inlier within all learned bounds
            return None
        if not self._cooldown_ok(state, "anomalous_movement", now):
            return None

        if violations:
            confidence = min(1.0, 0.7 + 0.1 * len(violations))
            details = (
                "outside observed range: "
                + ", ".join(f"{col}={features[col]:.2f}" for col in violations)
            )
        else:
            confidence = min(1.0, 0.6 + (-score) * 2.0)
            details = f"isolation-forest score {score:.3f}"
        return self._event("anomalous_movement", obj, confidence, details)

    def _record_sample(self, category: str, features: dict) -> None:
        history = self._history[category]
        history.append([features[c] for c in FEATURE_COLUMNS])
        if len(history) > self._max_history:
            del history[: len(history) - self._max_history]

        self._samples_since_fit[category] += 1
        if (
            len(history) >= self._min_samples
            and self._samples_since_fit[category] >= self._refit_interval
        ):
            self._fit(category)

    def _fit(self, category: str) -> None:
        frame = pd.DataFrame(self._history[category], columns=FEATURE_COLUMNS)
        model = IsolationForest(
            n_estimators=100, contamination=0.02, random_state=0
        )
        model.fit(frame)
        self._models[category] = model

        # Robust per-feature bounds: quantiles widened by IQR (or by the
        # observed magnitude when the history has almost no variance)
        bounds: dict[str, tuple[float, float]] = {}
        quantiles = frame[RANGE_GUARD_COLUMNS].quantile([0.01, 0.25, 0.75, 0.99])
        for col in RANGE_GUARD_COLUMNS:
            q01 = float(quantiles.loc[0.01, col])
            q25 = float(quantiles.loc[0.25, col])
            q75 = float(quantiles.loc[0.75, col])
            q99 = float(quantiles.loc[0.99, col])
            margin = max(3.0 * (q75 - q25), abs(q99), 1e-6)
            bounds[col] = (q01 - margin, q99 + margin)
        self._bounds[category] = bounds

        self._samples_since_fit[category] = 0
        logger.info(
            "BehaviorAnalyzer: fitted %s anomaly model on %d samples",
            category, len(frame),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _event(event_type: str, obj: TrackedObject, score: float, details: str) -> dict:
        return {
            "type": event_type,
            "track_id": obj.track_id,
            "category": obj.category,
            "score": round(score, 3),
            "details": details,
        }

    def _cooldown_ok(self, state: _TrackState, event_type: str, now: float) -> bool:
        last = state.last_event_times.get(event_type)
        if last is not None and now - last < self._event_cooldown:
            return False
        state.last_event_times[event_type] = now
        return True

    def _prune_stale_tracks(self, now: float) -> None:
        stale = [
            key
            for key, state in self._tracks.items()
            if now - state.last_seen > self._track_ttl
        ]
        for key in stale:
            del self._tracks[key]
