"""Runtime configuration, sourced from environment variables with sane defaults.

Kept as a tiny plain dataclass (rather than pulling in pydantic-settings) since
there are only a handful of knobs and all of them are optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: torchvision detection models this service knows how to build. Both are
#: COCO-pretrained and small enough to run comfortably on a CPU.
SUPPORTED_MODELS = (
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "ssdlite320_mobilenet_v3_large",
)

DEFAULT_MODEL_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"
DEFAULT_SCORE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Settings:
    """Service configuration, read once at process start."""

    model_name: str = DEFAULT_MODEL_NAME
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    device: str = "cpu"


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from environment variables.

    Environment variables:
        VISION_MODEL_NAME: one of ``SUPPORTED_MODELS``
            (default: fasterrcnn_mobilenet_v3_large_320_fpn).
        VISION_SCORE_THRESHOLD: float in [0, 1] below which detections
            are dropped (default: 0.5).
        VISION_DEVICE: torch device string, e.g. "cpu" or "cuda" (default: cpu).
    """
    model_name = os.environ.get("VISION_MODEL_NAME", DEFAULT_MODEL_NAME)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported VISION_MODEL_NAME={model_name!r}; must be one of {SUPPORTED_MODELS}"
        )
    score_threshold = float(os.environ.get("VISION_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD))
    device = os.environ.get("VISION_DEVICE", "cpu")
    return Settings(model_name=model_name, score_threshold=score_threshold, device=device)
