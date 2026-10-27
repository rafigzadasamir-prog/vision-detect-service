"""Generates the synthetic PNG fixtures used by the test suite and demo.

These images are drawn from scratch with PIL (simple flat-color shapes, text,
and procedurally generated noise) specifically so there is zero licensing
ambiguity -- no photograph, stock asset, or third-party content is used
anywhere in this repository.

Re-run with ``python tests/fixtures/generate_fixtures.py`` to regenerate.
"""

from __future__ import annotations

import pathlib
import random

from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = pathlib.Path(__file__).parent


def make_shapes_image() -> Image.Image:
    """Flat-color rectangles, an ellipse, and a triangle on a white canvas."""
    image = Image.new("RGB", (640, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.rectangle([40, 40, 260, 220], fill=(220, 60, 60), outline=(0, 0, 0), width=2)
    draw.ellipse([320, 60, 560, 260], fill=(60, 120, 220), outline=(0, 0, 0), width=2)
    draw.polygon([(100, 300), (260, 300), (180, 440)], fill=(60, 180, 90), outline=(0, 0, 0))
    draw.rectangle([350, 300, 600, 420], fill=(240, 180, 40), outline=(0, 0, 0), width=2)

    return image


def make_text_image() -> Image.Image:
    """A caption plus a couple of shapes, used to sanity-check PNG round-tripping."""
    image = Image.new("RGB", (480, 320), color=(30, 30, 40))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((20, 20), "VISION DETECT SERVICE", fill=(255, 255, 255), font=font)
    draw.text((20, 40), "synthetic test fixture - no real photo", fill=(200, 200, 200), font=font)
    draw.rounded_rectangle([40, 90, 200, 230], radius=12, fill=(90, 200, 160))
    draw.ellipse([260, 90, 420, 230], fill=(200, 90, 160))

    return image


def make_noise_image() -> Image.Image:
    """Procedurally generated grid of randomly colored tiles (deterministic seed)."""
    rng = random.Random(42)
    tile = 20
    width, height = 320, 240
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            draw.rectangle([x, y, x + tile, y + tile], fill=color)

    return image


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    make_shapes_image().save(FIXTURES_DIR / "shapes.png")
    make_text_image().save(FIXTURES_DIR / "text_and_shapes.png")
    make_noise_image().save(FIXTURES_DIR / "noise.png")
    print(f"Wrote fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
