FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir \
    matplotlib \
    nibabel \
    numpy==1.22.4 \
    pandas \
    Pillow \
    "protobuf>=3.20,<6" \
    pydicom \
    pylibjpeg==1.4.0 \
    pylibjpeg-libjpeg==1.3.4 \
    pylibjpeg-openjpeg==1.3.2 \
    scikit-image==0.20.0 \
    scikit-learn \
    scipy \
    tqdm \
    opencv-python-headless==4.7.0.68 \
    streamlit \
    streamlit-drawable-canvas-bogaczm  \
    imageio


RUN apt-get update && \
    apt-get install -y --no-install-recommends fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*
