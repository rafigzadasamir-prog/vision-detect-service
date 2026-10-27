from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vision_detect_service.detector import RawDetection, StubDetector
from vision_detect_service.main import app, get_detector

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

#: A fixed, hand-chosen set of detections so response-schema assertions can
#: pin exact values instead of merely checking shapes.
FIXED_DETECTIONS = [
    RawDetection(label="cat", confidence=0.95, box=(10.0, 20.0, 110.0, 220.0)),
    RawDetection(label="dog", confidence=0.60, box=(200.0, 50.0, 300.0, 150.0)),
]


@pytest.fixture
def fixed_detections() -> list[RawDetection]:
    return FIXED_DETECTIONS


@pytest.fixture
def stub_detector(fixed_detections: list[RawDetection]) -> StubDetector:
    return StubDetector(detections=fixed_detections)


@pytest.fixture
def client(stub_detector: StubDetector) -> TestClient:
    """A TestClient wired to the deterministic stub detector, not the real model."""
    app.dependency_overrides[get_detector] = lambda: stub_detector
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_detector, None)


@pytest.fixture
def shapes_image_path() -> pathlib.Path:
    return FIXTURES_DIR / "shapes.png"


@pytest.fixture
def text_image_path() -> pathlib.Path:
    return FIXTURES_DIR / "text_and_shapes.png"


@pytest.fixture
def noise_image_path() -> pathlib.Path:
    return FIXTURES_DIR / "noise.png"


@pytest.fixture
def shapes_image(shapes_image_path: pathlib.Path) -> Image.Image:
    with Image.open(shapes_image_path) as img:
        return img.convert("RGB")
