"""Unit tests for the pydantic schemas that define the API's response contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vision_detect_service.schemas import BoundingBox, Detection, DetectionResponse


def test_bounding_box_accepts_valid_coordinates() -> None:
    box = BoundingBox(x_min=1.0, y_min=2.0, x_max=50.0, y_max=60.0)
    assert box.x_min == 1.0
    assert box.y_max == 60.0


def test_bounding_box_rejects_negative_coordinates() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x_min=-1.0, y_min=0.0, x_max=10.0, y_max=10.0)


def test_detection_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        Detection(
            label="cat",
            confidence=1.5,
            box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
        )


def test_detection_rejects_confidence_below_zero() -> None:
    with pytest.raises(ValidationError):
        Detection(
            label="cat",
            confidence=-0.1,
            box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
        )


def test_detection_response_round_trips_through_json() -> None:
    response = DetectionResponse(
        detections=[
            Detection(
                label="dog",
                confidence=0.87,
                box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
            )
        ],
        image_width=640,
        image_height=480,
        model_name="stub",
    )
    payload = response.model_dump_json()
    restored = DetectionResponse.model_validate_json(payload)
    assert restored == response


def test_detection_response_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValidationError):
        DetectionResponse(detections=[], image_width=0, image_height=480, model_name="stub")
