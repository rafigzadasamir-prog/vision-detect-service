"""Drawing of detection results onto an image for the ``/detect/visualize`` demo endpoint."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from vision_detect_service.detector import RawDetection

# A small, stable palette so repeated labels get a consistent, distinguishable
# color without pulling in a colormap dependency.
_PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 165, 0),
    (0, 130, 200),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
]


def _color_for_label(label: str) -> tuple[int, int, int]:
    return _PALETTE[hash(label) % len(_PALETTE)]


def draw_detections(image: Image.Image, detections: list[RawDetection]) -> Image.Image:
    """Return a copy of ``image`` with bounding boxes and labels drawn on it."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for det in detections:
        x_min, y_min, x_max, y_max = det.box
        color = _color_for_label(det.label)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        caption = f"{det.label} {det.confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        label_top = max(y_min - text_height - 4, 0)
        draw.rectangle(
            [x_min, label_top, x_min + text_width + 4, label_top + text_height + 4],
            fill=color,
        )
        draw.text((x_min + 2, label_top + 1), caption, fill=(255, 255, 255), font=font)

    return annotated


def encode_png(image: Image.Image) -> bytes:
    """Serialize a PIL image to PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
