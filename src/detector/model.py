"""Violence detection model: loads a 3D CNN and classifies video clips."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from src.detector.preprocessing import preprocess_clip

logger = logging.getLogger(__name__)

LABELS = ("violence", "normal")


class ViolenceDetector:
    """Wraps a 3D CNN for binary violence / normal classification.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` weights file. If the file does not exist the
        model is initialised with ImageNet-pretrained weights and a
        randomly initialised classification head.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"`` (default) which picks CUDA
        when available.
    """

    def __init__(
        self,
        model_path: str = "models/violence_detector.pt",
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._device = self._resolve_device(device)
        self._model_name = "r3d_18"
        self._model = self._build_model()
        self.load_model()

    # -- Public API -----------------------------------------------------------

    def load_model(self) -> None:
        """Load weights from *model_path* if the file exists."""
        path = Path(self._model_path)
        if path.is_file():
            logger.info("Loading model weights from %s", path)
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
        else:
            logger.info(
                "Model file %s not found; using pretrained backbone with "
                "random classification head.",
                path,
            )
        self._model.to(self._device)
        self._model.eval()

    def predict(self, clip: np.ndarray) -> tuple[str, float]:
        """Classify a raw BGR clip as violence or normal.

        Parameters
        ----------
        clip:
            Numpy array of shape ``(T, H, W, 3)`` with uint8 BGR values
            (as returned by :pymethod:`Camera.get_clip`).

        Returns
        -------
        ``(label, confidence)`` where *label* is ``"violence"`` or
        ``"normal"`` and *confidence* is a float in ``[0, 1]``.
        """
        tensor = preprocess_clip(clip, device=self._device)

        with torch.no_grad():
            logits = self._model(tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=1)

        confidence, idx = probs.max(dim=1)
        label = LABELS[idx.item()]
        return label, round(confidence.item(), 6)

    # -- Properties -----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The device the model lives on."""
        return self._device

    @property
    def model_name(self) -> str:
        """Architecture name."""
        return self._model_name

    # -- Internals ------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_model(self) -> nn.Module:
        """Construct r3d_18 with a 2-class head."""
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(LABELS))
        return model
