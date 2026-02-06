"""Clip preprocessing pipeline for the violence detection model."""

from __future__ import annotations

import cv2
import numpy as np
import torch

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

INPUT_SIZE = 224


def _detect_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_clip(
    clip: np.ndarray,
    input_size: int = INPUT_SIZE,
    normalize: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a raw BGR clip from OpenCV into a model-ready tensor.

    Parameters
    ----------
    clip:
        Numpy array of shape (T, H, W, C) with uint8 BGR pixel values.
    input_size:
        Spatial dimension to resize frames to (square).
    normalize:
        If True, apply ImageNet mean/std normalization. If False, only
        scale to [0, 1].
    device:
        Target torch device. Auto-detected if None.

    Returns
    -------
    Tensor of shape (1, 3, T, input_size, input_size) on the target device.
    """
    if device is None:
        device = _detect_device()

    num_frames = clip.shape[0]
    frames: list[np.ndarray] = []

    for i in range(num_frames):
        frame = clip[i]
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to (input_size, input_size)
        frame = cv2.resize(frame, (input_size, input_size))
        frames.append(frame)

    # Stack: (T, H, W, C)
    arr = np.stack(frames).astype(np.float32) / 255.0

    if normalize:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
        arr = (arr - mean) / std

    # (T, H, W, C) -> (C, T, H, W)
    arr = arr.transpose(3, 0, 1, 2)

    # Add batch dimension: (1, C, T, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0)

    return tensor.to(device)
