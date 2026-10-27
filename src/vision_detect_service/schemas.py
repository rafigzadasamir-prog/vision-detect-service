"""Pydantic request/response schemas defining the public API contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates of the original image."""

    x_min: float = Field(..., ge=0, description="Left edge, in pixels.")
    y_min: float = Field(..., ge=0, description="Top edge, in pixels.")
    x_max: float = Field(..., ge=0, description="Right edge, in pixels.")
    y_max: float = Field(..., ge=0, description="Bottom edge, in pixels.")


class Detection(BaseModel):
    """A single detected object."""

    label: str = Field(..., description="Human-readable COCO class name, e.g. 'dog'.")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score.")
    box: BoundingBox


class DetectionResponse(BaseModel):
    """Response body for ``POST /detect``."""

    detections: list[Detection]
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    model_name: str = Field(..., description="Name of the model that produced these detections.")


class ErrorResponse(BaseModel):
    """Response body for error conditions (invalid upload, etc.)."""

    detail: str
