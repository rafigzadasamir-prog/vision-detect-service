"""FastAPI application exposing the object detection HTTP API.

Request flow for both endpoints::

    client upload (multipart/form-data)
        -> validate content-type / decode bytes as an image   (preprocess)
        -> ObjectDetector.detect(image)                        (model)
        -> filter by score threshold, map to response schema   (postprocess)
        -> JSON (/detect) or annotated PNG (/detect/visualize) (response)
"""

from __future__ import annotations

import io
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from vision_detect_service.config import get_settings
from vision_detect_service.detector import ObjectDetector, RawDetection, TorchvisionDetector
from vision_detect_service.schemas import BoundingBox, Detection, DetectionResponse
from vision_detect_service.visualize import draw_detections, encode_png

app = FastAPI(
    title="vision-detect-service",
    description="Object detection microservice built on a pretrained torchvision model.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def _build_default_detector() -> ObjectDetector:
    """Build the real torchvision-backed detector, once per process.

    Lazily constructed (rather than at import time) so that importing
    ``main`` for schema generation or in tests that override this dependency
    never triggers a model build or a weight download.
    """
    settings = get_settings()
    return TorchvisionDetector(
        model_name=settings.model_name,
        score_threshold=settings.score_threshold,
        device=settings.device,
    )


def get_detector() -> ObjectDetector:
    """FastAPI dependency providing the detection backend.

    Overridden in tests via ``app.dependency_overrides[get_detector]`` to
    inject a fast, deterministic stub instead of the real model.
    """
    return _build_default_detector()


async def _load_image(file: UploadFile) -> Image.Image:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type!r}. Expected an image.",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    return image.convert("RGB")


def _to_response(
    detections: list[RawDetection], image: Image.Image, model_name: str
) -> DetectionResponse:
    width, height = image.size
    return DetectionResponse(
        detections=[
            Detection(
                label=det.label,
                confidence=det.confidence,
                box=BoundingBox(
                    x_min=det.box[0], y_min=det.box[1], x_max=det.box[2], y_max=det.box[3]
                ),
            )
            for det in detections
        ],
        image_width=width,
        image_height=height,
        model_name=model_name,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Does not touch the model, so it stays fast."""
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile,
    detector: ObjectDetector = Depends(get_detector),
) -> DetectionResponse:
    """Run object detection on an uploaded image and return structured JSON."""
    image = await _load_image(file)
    detections = detector.detect(image)
    model_name = getattr(detector, "model_name", detector.__class__.__name__)
    return _to_response(detections, image, model_name)


@app.post(
    "/detect/visualize",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def detect_visualize(
    file: UploadFile,
    detector: ObjectDetector = Depends(get_detector),
) -> Response:
    """Run object detection and return the image with bounding boxes drawn on it (PNG)."""
    image = await _load_image(file)
    detections = detector.detect(image)
    annotated = draw_detections(image, detections)
    return Response(content=encode_png(annotated), media_type="image/png")
