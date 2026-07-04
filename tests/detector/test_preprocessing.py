"""Tests for clip preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.detector.preprocessing import preprocess_clip


@pytest.fixture()
def fake_clip() -> np.ndarray:
    """Create a synthetic BGR clip: (16, 480, 640, 3) uint8."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(16, 480, 640, 3), dtype=np.uint8)


class TestPreprocessClipOutputShape:
    """Verify the tensor shape is (1, C, T, H, W)."""

    def test_preprocess_clip_output_shape(self, fake_clip: np.ndarray) -> None:
        tensor = preprocess_clip(fake_clip)
        # batch=1, channels=3, T=16, H=224, W=224
        assert tensor.shape == (1, 3, 16, 224, 224)

    def test_preprocess_clip_custom_clip_length(self) -> None:
        rng = np.random.default_rng(0)
        clip = rng.integers(0, 256, size=(8, 240, 320, 3), dtype=np.uint8)
        tensor = preprocess_clip(clip)
        assert tensor.shape == (1, 3, 8, 224, 224)


class TestPreprocessClipOutputDtype:
    """Verify the output is a float tensor."""

    def test_preprocess_clip_output_dtype(self, fake_clip: np.ndarray) -> None:
        tensor = preprocess_clip(fake_clip)
        assert tensor.dtype == torch.float32


class TestPreprocessClipRgbConversion:
    """Verify BGR -> RGB conversion happens."""

    def test_preprocess_clip_rgb_conversion(self) -> None:
        # Create a clip where BGR channel values are distinct and known
        # Frame: all pixels have B=100, G=150, R=200
        frame = np.full((224, 224, 3), fill_value=0, dtype=np.uint8)
        frame[:, :, 0] = 100  # B
        frame[:, :, 1] = 150  # G
        frame[:, :, 2] = 200  # R
        clip = np.stack([frame] * 16)  # (16, 224, 224, 3)

        tensor = preprocess_clip(clip, normalize=False)
        # After BGR->RGB and no normalization, channels should be R, G, B
        # tensor shape: (1, 3, 16, 224, 224)
        # Channel 0 = R = 200/255, Channel 1 = G = 150/255, Channel 2 = B = 100/255
        r_val = tensor[0, 0, 0, 0, 0].item()
        g_val = tensor[0, 1, 0, 0, 0].item()
        b_val = tensor[0, 2, 0, 0, 0].item()
        assert pytest.approx(r_val, abs=0.01) == 200 / 255
        assert pytest.approx(g_val, abs=0.01) == 150 / 255
        assert pytest.approx(b_val, abs=0.01) == 100 / 255


class TestPreprocessClipNormalizationRange:
    """Verify pixel values are in a reasonable normalized range."""

    def test_preprocess_clip_normalization_range(self, fake_clip: np.ndarray) -> None:
        tensor = preprocess_clip(fake_clip)
        # With ImageNet normalization, values typically range ~[-2.5, 2.5]
        assert tensor.min().item() >= -5.0
        assert tensor.max().item() <= 5.0

    def test_preprocess_clip_no_normalize_range(self, fake_clip: np.ndarray) -> None:
        tensor = preprocess_clip(fake_clip, normalize=False)
        # Without normalization, values should be in [0, 1]
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0
