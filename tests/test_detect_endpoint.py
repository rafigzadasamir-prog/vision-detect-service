"""API contract tests for POST /detect, run against the deterministic stub detector."""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from vision_detect_service.detector import RawDetection, StubDetector
from vision_detect_service.main import app, get_detector


def _post_image(client: TestClient, image_path: pathlib.Path, endpoint: str = "/detect"):
    with image_path.open("rb") as fh:
        return client.post(endpoint, files={"file": (image_path.name, fh, "image/png")})


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_returns_200_and_matches_stub_detections(
    client: TestClient,
    shapes_image_path: pathlib.Path,
    fixed_detections: list[RawDetection],
) -> None:
    response = _post_image(client, shapes_image_path)
    assert response.status_code == 200

    body = response.json()
    assert body["image_width"] == 640
    assert body["image_height"] == 480
    assert len(body["detections"]) == len(fixed_detections)

    first = body["detections"][0]
    expected = fixed_detections[0]
    assert first["label"] == expected.label
    assert first["confidence"] == expected.confidence
    assert first["box"] == {
        "x_min": expected.box[0],
        "y_min": expected.box[1],
        "x_max": expected.box[2],
        "y_max": expected.box[3],
    }


def test_detect_response_matches_schema_contract(
    client: TestClient, text_image_path: pathlib.Path
) -> None:
    response = _post_image(client, text_image_path)
    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) == {"detections", "image_width", "image_height", "model_name"}
    for detection in body["detections"]:
        assert set(detection.keys()) == {"label", "confidence", "box"}
        assert isinstance(detection["label"], str)
        assert 0.0 <= detection["confidence"] <= 1.0
        box = detection["box"]
        assert set(box.keys()) == {"x_min", "y_min", "x_max", "y_max"}
        assert box["x_min"] < box["x_max"]
        assert box["y_min"] < box["y_max"]


def test_detect_works_on_all_sample_fixtures(
    client: TestClient,
    shapes_image_path: pathlib.Path,
    text_image_path: pathlib.Path,
    noise_image_path: pathlib.Path,
) -> None:
    for path in (shapes_image_path, text_image_path, noise_image_path):
        response = _post_image(client, path)
        assert response.status_code == 200, f"failed for {path.name}"
        assert response.json()["detections"], f"expected detections for {path.name}"


def test_detect_rejects_non_image_upload(client: TestClient) -> None:
    response = client.post(
        "/detect", files={"file": ("not_an_image.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]


def test_detect_rejects_corrupt_image_bytes(client: TestClient) -> None:
    response = client.post(
        "/detect",
        files={"file": ("broken.png", b"\x89PNGnot-actually-png-data", "image/png")},
    )
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]


def test_detect_rejects_empty_upload(client: TestClient) -> None:
    response = client.post("/detect", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_detect_requires_file_field(client: TestClient) -> None:
    response = client.post("/detect")
    assert response.status_code == 422


def test_detect_with_zero_detections_returns_empty_list(
    client: TestClient, shapes_image_path: pathlib.Path
) -> None:
    app.dependency_overrides[get_detector] = lambda: StubDetector(detections=[])
    try:
        response = _post_image(client, shapes_image_path)
    finally:
        app.dependency_overrides.pop(get_detector, None)

    assert response.status_code == 200
    assert response.json()["detections"] == []
