"""Tiny local demo UI for vision-detect-service.

Run the FastAPI service first, then:

    streamlit run app_streamlit.py

Upload an image and see the annotated detection result returned by the
``/detect/visualize`` endpoint, plus the raw JSON from ``/detect``.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("VISION_API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Vision Detect Service Demo", page_icon=":mag:")
st.title("Vision Detect Service — Demo")
st.caption(f"Calling API at {API_BASE_URL}")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type or "image/png")}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image_bytes, use_container_width=True)

    with st.spinner("Running detection..."):
        detect_response = requests.post(f"{API_BASE_URL}/detect", files=files, timeout=60)
        viz_response = requests.post(f"{API_BASE_URL}/detect/visualize", files=files, timeout=60)

    if detect_response.status_code != 200 or viz_response.status_code != 200:
        st.error(
            f"API request failed: detect={detect_response.status_code}, "
            f"visualize={viz_response.status_code}"
        )
    else:
        with col2:
            st.subheader("Detections")
            st.image(viz_response.content, use_container_width=True)

        st.subheader("Raw JSON response")
        st.json(detect_response.json())
else:
    st.info("Upload an image to run detection, or try one of the sample images in tests/fixtures/.")
