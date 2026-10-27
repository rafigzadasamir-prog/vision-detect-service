"""Object detection backends.

``ObjectDetector`` is a small protocol so that the FastAPI layer never talks
to torchvision directly. This keeps the real model swappable for a fast,
dependency-free stub in tests -- unit tests exercise the full HTTP contract
(validation, response schema, image encoding) without downloading or running
a neural network, while ``@pytest.mark.integration`` tests cover the real
``TorchvisionDetector`` end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class RawDetection:
    """A single detection in a backend-agnostic form."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in pixels


class ObjectDetector(Protocol):
    """Interface every detection backend must satisfy."""

    def detect(self, image: Image.Image) -> list[RawDetection]:
        """Run detection on a single PIL image and return raw detections."""
        ...


class TorchvisionDetector:
    """Wraps a COCO-pretrained torchvision detection model.

    Weights are downloaded automatically by torchvision on first use (cached
    under ``~/.cache/torch/hub/checkpoints``) -- there is no bundled weight
    file in this repository. The first call to :meth:`detect` (or service
    startup, since the model is built eagerly) therefore requires network
    access; subsequent runs reuse the cache and work fully offline.
    """

    def __init__(
        self,
        model_name: str = "fasterrcnn_mobilenet_v3_large_320_fpn",
        score_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        # Imported lazily so that importing this module (e.g. for the stub
        # detector used by most tests) never pulls torch/torchvision in.
        import torch
        import torchvision

        self._torch = torch
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.device = device

        # torchvision's model registry resolves the builder function and its
        # matching Weights enum from the same string name, so model naming
        # conventions (e.g. "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights")
        # never need to be duplicated here.
        try:
            weights_enum = torchvision.models.get_model_weights(model_name)
        except ValueError as exc:
            raise ValueError(f"Unknown torchvision detection model: {model_name!r}") from exc
        weights = weights_enum.DEFAULT

        self._categories: list[str] = weights.meta["categories"]
        self._preprocess = weights.transforms()

        model = torchvision.models.get_model(model_name, weights=weights)
        model.eval()
        self._model = model.to(device)

    def detect(self, image: Image.Image) -> list[RawDetection]:
        rgb_image = image.convert("RGB")
        input_tensor = self._preprocess(rgb_image).to(self.device)

        with self._torch.no_grad():
            output = self._model([input_tensor])[0]

        detections: list[RawDetection] = []
        boxes = output["boxes"].tolist()
        labels = output["labels"].tolist()
        scores = output["scores"].tolist()
        for box, label_idx, score in zip(boxes, labels, scores, strict=True):
            if score < self.score_threshold:
                continue
            x_min, y_min, x_max, y_max = box
            detections.append(
                RawDetection(
                    label=self._categories[label_idx],
                    confidence=float(score),
                    box=(float(x_min), float(y_min), float(x_max), float(y_max)),
                )
            )
        return detections


class StubDetector:
    """Deterministic, dependency-free detector used by default in tests.

    Returns a fixed set of detections regardless of input, scaled to fit
    within the given image's bounds so visualization logic can be exercised
    without ever loading torch/torchvision.
    """

    def __init__(self, detections: list[RawDetection] | None = None) -> None:
        self._fixed_detections = detections

    def detect(self, image: Image.Image) -> list[RawDetection]:
        if self._fixed_detections is not None:
            return self._fixed_detections

        width, height = image.size
        return [
            RawDetection(
                label="cat",
                confidence=0.92,
                box=(width * 0.1, height * 0.1, width * 0.5, height * 0.5),
            ),
            RawDetection(
                label="dog",
                confidence=0.81,
                box=(width * 0.55, height * 0.4, width * 0.95, height * 0.9),
            ),
        ]
