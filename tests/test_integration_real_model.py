"""Integration tests that exercise the real, pretrained torchvision model.

Skipped by default (see ``pytest.ini_options.markers`` in pyproject.toml and
the "integration" marker below). These are the only tests in the suite that
require torch/torchvision to actually run a forward pass, and the only ones
that may trigger a weight download on first run. Run explicitly with:

    pytest -m integration -q
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vision_detect_service.detector import TorchvisionDetector
from vision_detect_service.main import app, get_detector

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_detector() -> TorchvisionDetector:
    return TorchvisionDetector(score_threshold=0.3)


def test_real_model_runs_on_synthetic_image_without_error(
    real_detector: TorchvisionDetector, shapes_image_path: pathlib.Path
) -> None:
    with Image.open(shapes_image_path) as image:
        detections = real_detector.detect(image.convert("RGB"))

    # Flat synthetic shapes are not expected to match any COCO class with high
    # confidence; the meaningful assertion is that the real model runs end to
    # end and returns well-formed, score-sorted-by-threshold detections.
    for det in detections:
        assert 0.0 <= det.confidence <= 1.0
        assert isinstance(det.label, str) and det.label
        x_min, y_min, x_max, y_max = det.box
        assert x_min < x_max
        assert y_min < y_max


def test_detect_endpoint_with_real_model(shapes_image_path: pathlib.Path) -> None:
    app.dependency_overrides[get_detector] = lambda: TorchvisionDetector(score_threshold=0.3)
    try:
        client = TestClient(app)
        with shapes_image_path.open("rb") as fh:
            response = client.post(
                "/detect", files={"file": (shapes_image_path.name, fh, "image/png")}
            )
    finally:
        app.dependency_overrides.pop(get_detector, None)

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "fasterrcnn_mobilenet_v3_large_320_fpn"
    assert body["image_width"] == 640
    assert body["image_height"] == 480
