"""Tests for the ViolenceDetector model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.detector.model import ViolenceDetector


@pytest.fixture()
def fake_clip() -> np.ndarray:
    """Create a synthetic BGR clip: (16, 480, 640, 3) uint8."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(16, 480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def detector() -> ViolenceDetector:
    """Create a ViolenceDetector with default (no weights file) on CPU."""
    return ViolenceDetector(model_path="nonexistent_model.pt", device="cpu")


class TestDetectorInitialization:
    """Test detector construction and device selection."""

    def test_detector_initializes_on_cpu(self, detector: ViolenceDetector) -> None:
        assert detector.device == torch.device("cpu")

    def test_detector_model_name(self, detector: ViolenceDetector) -> None:
        assert detector.model_name == "r3d_18"

    def test_detector_model_is_in_eval_mode(self, detector: ViolenceDetector) -> None:
        assert not detector._model.training


class TestDetectorPredict:
    """Test prediction output format and value ranges."""

    def test_detector_predict_returns_label_and_confidence(
        self, detector: ViolenceDetector, fake_clip: np.ndarray
    ) -> None:
        label, confidence = detector.predict(fake_clip)
        assert isinstance(label, str)
        assert isinstance(confidence, float)

    def test_detector_predict_confidence_range(
        self, detector: ViolenceDetector, fake_clip: np.ndarray
    ) -> None:
        _, confidence = detector.predict(fake_clip)
        assert 0.0 <= confidence <= 1.0

    def test_detector_predict_returns_violence_or_normal(
        self, detector: ViolenceDetector, fake_clip: np.ndarray
    ) -> None:
        label, _ = detector.predict(fake_clip)
        assert label in ("violence", "normal")

    def test_detector_predict_with_mocked_model(self, fake_clip: np.ndarray) -> None:
        """Use a mock to force a specific prediction."""
        det = ViolenceDetector(model_path="nonexistent.pt", device="cpu")
        # Mock the model to return known logits: class 0 (violence) > class 1 (normal)
        fake_output = torch.tensor([[2.0, -1.0]])
        det._model = MagicMock(return_value=fake_output)
        det._model.training = False

        label, confidence = det.predict(fake_clip)
        assert label == "violence"
        assert confidence > 0.5

    def test_detector_predict_normal_with_mocked_model(
        self, fake_clip: np.ndarray
    ) -> None:
        """Use a mock to force a normal prediction."""
        det = ViolenceDetector(model_path="nonexistent.pt", device="cpu")
        # Mock: class 1 (normal) > class 0 (violence)
        fake_output = torch.tensor([[-2.0, 3.0]])
        det._model = MagicMock(return_value=fake_output)
        det._model.training = False

        label, confidence = det.predict(fake_clip)
        assert label == "normal"
        assert confidence > 0.5
