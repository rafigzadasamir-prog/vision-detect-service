"""Tests for bounding-box visualization logic and the /detect/visualize endpoint."""

from __future__ import annotations

import io
import pathlib

from fastapi.testclient import TestClient
from PIL import Image

from vision_detect_service.detector import RawDetection
from vision_detect_service.visualize import draw_detections, encode_png


def test_draw_detections_returns_same_size_image(shapes_image: Image.Image) -> None:
    detections = [RawDetection(label="cat", confidence=0.9, box=(10, 10, 100, 100))]
    annotated = draw_detections(shapes_image, detections)
    assert annotated.size == shapes_image.size
    assert annotated.mode == "RGB"


def test_draw_detections_modifies_pixels_within_box(shapes_image: Image.Image) -> None:
    # Pick a region that is plain white in the source fixture (top-right
    # corner, outside every drawn shape) and draw a box around it; the
    # outline pixels should no longer be pure white afterwards.
    box = (580.0, 10.0, 630.0, 50.0)
    assert shapes_image.getpixel((600, 20)) == (255, 255, 255)

    detections = [RawDetection(label="dog", confidence=0.75, box=box)]
    annotated = draw_detections(shapes_image, detections)

    x_min, y_min, x_max, y_max = (int(v) for v in box)
    edge_pixels = [
        annotated.getpixel((x_min + 1, y_min + 1)),
        annotated.getpixel((x_max - 1, y_min + 1)),
    ]
    assert any(pixel != (255, 255, 255) for pixel in edge_pixels)


def test_draw_detections_with_no_boxes_returns_unmodified_copy(shapes_image: Image.Image) -> None:
    annotated = draw_detections(shapes_image, [])
    assert list(annotated.getdata()) == list(shapes_image.convert("RGB").getdata())
    assert annotated is not shapes_image


def test_draw_detections_handles_box_touching_image_edge(shapes_image: Image.Image) -> None:
    width, height = shapes_image.size
    detections = [RawDetection(label="edge", confidence=0.5, box=(0.0, 0.0, width, height))]
    # Must not raise even when the label caption would be drawn above y=0.
    annotated = draw_detections(shapes_image, detections)
    assert annotated.size == (width, height)


def test_encode_png_round_trips(shapes_image: Image.Image) -> None:
    png_bytes = encode_png(shapes_image)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    decoded = Image.open(io.BytesIO(png_bytes))
    assert decoded.size == shapes_image.size


def test_visualize_endpoint_returns_png(
    client: TestClient, shapes_image_path: pathlib.Path
) -> None:
    with shapes_image_path.open("rb") as fh:
        response = client.post(
            "/detect/visualize", files={"file": (shapes_image_path.name, fh, "image/png")}
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    decoded = Image.open(io.BytesIO(response.content))
    assert decoded.size == (640, 480)


def test_visualize_endpoint_rejects_non_image_upload(client: TestClient) -> None:
    response = client.post(
        "/detect/visualize", files={"file": ("not_an_image.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400
