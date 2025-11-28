# HuggingFace Spaces compatible Dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Run backend on port 7860 (HuggingFace standard)
CMD ["uvicorn", "deployment.backend.backend_amar:app", "--host", "0.0.0.0", "--port", "7860"]
