FROM python:3.11-slim

WORKDIR /app

# System libs required by Pillow/torchvision for image I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

ENV VISION_MODEL_NAME=fasterrcnn_mobilenet_v3_large_320_fpn \
    VISION_SCORE_THRESHOLD=0.5 \
    VISION_DEVICE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Pretrained COCO weights are downloaded automatically by torchvision on the
# first request (cached under /root/.cache/torch) -- the image itself ships
# no model weights, so the first container start needs network access.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "vision_detect_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
