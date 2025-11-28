# Base Image: NVIDIA CUDA 11.3 with cuDNN8 on Ubuntu 20.04
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3-distutils \
        python3-pip \
        git \
        build-essential \
        libsm6 \
        libxext6 \
        libgl1 \
        gfortran \
        libssl-dev \
        wget \
        curl && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip setuptools wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch 1.11.0 with CUDA 11.3
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install OpenMMLab dependencies
RUN pip install -U openmim && \
    mim install mmcv-full==1.5.0

# Install timm and mmdet
RUN pip install timm==0.6.11 mmdet==2.28.1

# Install utility libraries
RUN pip install Pillow==9.5.0 opencv-python termcolor yacs pyyaml scipy

# Install DCNv3 wheel (compatible with Python 3.8, Torch 1.11, CUDA 11.3)
RUN pip install https://github.com/OpenGVLab/InternImage/releases/download/whl_files/DCNv3-1.0+cu113torch1.11.0-cp38-cp38-linux_x86_64.whl

# Copy application code
COPY . /app/

# Install any Python dependencies from requirements.txt (if it exists)
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Expose ports for frontend (8080) and backend (8000)
EXPOSE 8000 8080

# Default command
CMD ["/bin/bash"]
