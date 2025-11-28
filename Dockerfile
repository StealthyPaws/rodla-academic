# HuggingFace Spaces compatible Dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for mmcv compilation and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        ninja-build \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libssl-dev \
        libcuda-dev \
        wget \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (excluding large files via .dockerignore)
COPY . /app/

# Create checkpoint directory
RUN mkdir -p /app/finetuning_rodla/finetuning_rodla/checkpoints

# Optional: Attempt to download weights at build time (non-blocking)
# If download fails or times out, the backend will fall back to heuristic detection
# Weights can be manually placed in: /app/finetuning_rodla/finetuning_rodla/checkpoints/rodla_internimage_xl_publaynet.pth
RUN timeout 300 python /app/deployment/backend/download_weights.py 2>&1 || echo "⚠️  Weight download timed out or failed - backend will use heuristic detection. To enable full detection, manually place weights file in checkpoints directory."

# Run backend.py on port 7860 (HuggingFace standard)
CMD ["python", "deployment/backend/backend.py"]
