"""Violence detection using a pre-trained ViT image classifier.

Uses the jaranohaal/vit-base-violence-detection model from HuggingFace.
Model weights are in timm format, preprocessing uses the transformers
ViTImageProcessor for correct normalization (no center-crop).

Label mapping: class 0 = normal, class 1 = violence.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_HF_ID = "jaranohaal/vit-base-violence-detection"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "vit-violence-detection"

VIOLENCE_CLASS_IDX = 1


class ViolenceDetector:
    """Wraps a ViT image classifier for violence detection.

    Classifies sampled frames from a video clip and averages their
    violence probability for a final prediction.

    Parameters
    ----------
    model_path:
        HuggingFace model ID. Defaults to the pre-trained violence
        detection model.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"`` (default).
    """

    def __init__(
        self,
        model_path: str = MODEL_HF_ID,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._device = self._resolve_device(device)
        self._model_name = "vit-violence-detection"
        # One model instance is shared by all camera pipelines
        self._infer_lock = threading.Lock()

        logger.info("Loading model %s on %s...", model_path, self._device)
        self._model, self._processor = self._load_model(model_path)
        self._model.to(self._device)
        self._model.eval()
        logger.info("Model loaded successfully.")

    def _load_model(self, model_path: str):
        """Load timm ViT model with transformers processor for preprocessing.

        Loads from local ``models/vit-violence-detection/`` directory if it
        exists, otherwise falls back to downloading from HuggingFace.
        """
        import timm
        from safetensors.torch import load_file
        from transformers import ViTImageProcessor

        # Prefer local model directory, fall back to HuggingFace
        if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "model.safetensors").exists():
            source = str(LOCAL_MODEL_DIR)
            weights_path = str(LOCAL_MODEL_DIR / "model.safetensors")
            logger.info("Loading model from local directory: %s", source)
        else:
            from huggingface_hub import hf_hub_download
            source = model_path
            weights_path = hf_hub_download(model_path, "model.safetensors")
            logger.info("Local model not found, downloading from HuggingFace: %s", model_path)

        # Create architecture and load weights (timm format)
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)

        # Use transformers processor for correct preprocessing
        # (direct resize to 224x224, no center-crop)
        processor = ViTImageProcessor.from_pretrained(source)

        return model, processor

    def predict(self, clip: np.ndarray) -> tuple[str, float]:
        """Classify a video clip by sampling frames.

        Parameters
        ----------
        clip:
            Numpy array of shape ``(T, H, W, 3)`` with uint8 BGR values.

        Returns
        -------
        ``(label, confidence)`` where *label* is ``"violence"`` or
        ``"normal"`` and *confidence* is a float in ``[0, 1]``.
        """
        n_frames = len(clip)
        sample_indices = np.linspace(0, n_frames - 1, min(4, n_frames), dtype=int)

        violence_scores = []

        for idx in sample_indices:
            frame_bgr = clip[idx]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            inputs = self._processor(images=image, return_tensors="pt")
            tensor = inputs["pixel_values"].to(self._device)

            with self._infer_lock, torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)

            violence_scores.append(probs[0, VIOLENCE_CLASS_IDX].item())

        avg_violence = sum(violence_scores) / len(violence_scores)

        if avg_violence >= 0.5:
            return "violence", round(avg_violence, 4)
        else:
            return "normal", round(1.0 - avg_violence, 4)

    def predict_frame(self, frame: np.ndarray) -> tuple[str, float]:
        """Classify a single BGR frame."""
        return self.predict(frame[np.newaxis, ...])

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
