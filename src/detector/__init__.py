"""Detector module: violence detection model and preprocessing."""

from src.detector.model import ViolenceDetector
from src.detector.preprocessing import preprocess_clip

__all__ = ["ViolenceDetector", "preprocess_clip"]
