"""Tests for the ViolenceDetector model.

Model loading is mocked so the tests run offline without downloading
weights from HuggingFace. The mocked model returns fixed logits for
class 0 = normal, class 1 = violence (see VIOLENCE_CLASS_IDX).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.detector.model import ViolenceDetector, VIOLENCE_CLASS_IDX


def _make_detector(logits: list[float]) -> ViolenceDetector:
    """Build a detector whose model always returns the given logits."""
    fake_model = MagicMock(return_value=torch.tensor([logits]))
    fake_processor = MagicMock(
        return_value={"pixel_values": torch.zeros(1, 3, 224, 224)}
    )
    with patch.object(
        ViolenceDetector, "_load_model", return_value=(fake_model, fake_processor)
    ):
        return ViolenceDetector(device="cpu")


@pytest.fixture()
def fake_clip() -> np.ndarray:
    """Create a synthetic BGR clip: (8, 480, 640, 3) uint8."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(8, 480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def fake_frame() -> np.ndarray:
    """Create a single synthetic BGR frame."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


class TestDetectorInitialization:
    """Test detector construction and device selection."""

    def test_detector_initializes_on_cpu(self) -> None:
        det = _make_detector([0.0, 0.0])
        assert det.device == torch.device("cpu")

    def test_detector_model_name(self) -> None:
        det = _make_detector([0.0, 0.0])
        assert det.model_name == "vit-violence-detection"

    def test_violence_class_index_is_one(self) -> None:
        # Class 0 = normal, class 1 = violence (per the HF model card)
        assert VIOLENCE_CLASS_IDX == 1

    def test_resolve_device_explicit_cpu(self) -> None:
        assert ViolenceDetector._resolve_device("cpu") == torch.device("cpu")

    def test_resolve_device_auto(self) -> None:
        resolved = ViolenceDetector._resolve_device("auto")
        assert resolved in (torch.device("cpu"), torch.device("cuda"))


class TestDetectorPredict:
    """Test prediction output format, value ranges, and label mapping."""

    def test_predict_returns_label_and_confidence(self, fake_clip: np.ndarray) -> None:
        det = _make_detector([0.5, 0.5])
        label, confidence = det.predict(fake_clip)
        assert isinstance(label, str)
        assert isinstance(confidence, float)

    def test_predict_confidence_range(self, fake_clip: np.ndarray) -> None:
        det = _make_detector([0.3, 1.2])
        _, confidence = det.predict(fake_clip)
        assert 0.0 <= confidence <= 1.0

    def test_predict_violence_when_class1_dominates(self, fake_clip: np.ndarray) -> None:
        det = _make_detector([-2.0, 3.0])  # class 1 (violence) wins
        label, confidence = det.predict(fake_clip)
        assert label == "violence"
        assert confidence > 0.5

    def test_predict_normal_when_class0_dominates(self, fake_clip: np.ndarray) -> None:
        det = _make_detector([3.0, -2.0])  # class 0 (normal) wins
        label, confidence = det.predict(fake_clip)
        assert label == "normal"
        assert confidence > 0.5

    def test_predict_frame_single_frame(self, fake_frame: np.ndarray) -> None:
        det = _make_detector([-1.0, 2.0])
        label, confidence = det.predict_frame(fake_frame)
        assert label == "violence"
        assert 0.0 <= confidence <= 1.0

    def test_predict_samples_at_most_four_frames(self, fake_clip: np.ndarray) -> None:
        det = _make_detector([0.0, 1.0])
        det.predict(fake_clip)
        # 8-frame clip should be sampled down to 4 inference calls
        assert det._model.call_count == 4
