"""Detector module: violence classification and people/vehicle tracking."""

from src.detector.model import ViolenceDetector
from src.detector.objects import ObjectTracker, TrackedObject

__all__ = ["ViolenceDetector", "ObjectTracker", "TrackedObject"]
