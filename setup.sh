#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
ENV_NAME="RoDLA"
ENV_PATH="./$ENV_NAME"

# URLs for PyTorch/Detectron2 wheels
TORCH_VERSION="1.11.0+cu113"
TORCH_URL="https://download.pytorch.org/whl/cu113/torch_stable.html"

DETECTRON2_VERSION="cu113/torch1.11"
DETECTRON2_URL="https://dl.fbaipublicfiles.com/detectron2/wheels/$DETECTRON2_VERSION/index.html"

DCNV3_URL="https://github.com/OpenGVLab/InternImage/releases/download/whl_files/DCNv3-1.0+cu113torch1.11.0-cp37-cp37m-linux_x86_64.whl"

# Check if the environment exists and activate it
if [ ! -d "$ENV_PATH" ]; then
    echo "❌ Error: Virtual environment '$ENV_NAME' not found at '$ENV_PATH'."
    echo "Please ensure you have created the environment using 'python3.7 -m venv $ENV_NAME' first."
    exit 1
fi

echo "--- 🛠️ Activating Virtual Environment: $ENV_NAME ---"
# Deactivate if active, then activate the target environment
# We use the full path to pip/python for reliability instead of 'source' which only affects the current shell session.
export PATH="$ENV_PATH/bin:$PATH"

# Check if the activation worked by checking the 'which python' command
if ! command -v python | grep -q "$ENV_PATH"; then
    echo "❌ Failed to set environment path. Aborting."
    exit 1
fi

echo "--- 🗑️ Uninstalling Old PyTorch Packages (if present) ---"
# Use the environment's pip (now in $PATH)
pip uninstall torch torchvision torchaudio -y || true

echo "--- 📦 Installing PyTorch 1.11.0+cu113 and Core Dependencies ---"
# Note: We are using the correct PyTorch 1.11.0 versions that match the DCNv3 wheel.
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f "$TORCH_URL"

echo "--- 📦 Installing OpenMMLab and Other Benchmarking Dependencies ---"
pip install -U openmim
# Ensure the full path to python is used for detectron2 (though it should be the venv python now)
python -m pip install detectron2 -f "$DETECTRON2_URL"
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
pip install Pillow==9.5.0
pip install opencv-python termcolor yacs pyyaml scipy

echo "--- 🚀 Installing Compatible DCNv3 Wheel ---"
pip install "$DCNV3_URL"

echo "--- ✅ Setup Complete ---"
echo "The $ENV_NAME environment is configured. To use it, run:"
echo "source $ENV_PATH/bin/activate"