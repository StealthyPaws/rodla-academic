#!/usr/bin/env python3
"""
Download model weights from Google Drive
This script is used during Docker build or startup to download required weights
"""
import os
import urllib.request
import sys
from pathlib import Path
import gdown

def download_weights():
    """Download weights from Google Drive"""
    
    # Google Drive file ID from the shared link
    # https://drive.google.com/file/d/1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz/view?usp=sharing
    DRIVE_FILE_ID = "1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz"
    
    # Destination path
    REPO_ROOT = Path("/home/admin/CV/rodla-academic")
    WEIGHTS_DIR = REPO_ROOT / "finetuning_rodla/finetuning_rodla/checkpoints"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    WEIGHTS_PATH = WEIGHTS_DIR / "rodla_internimage_xl_publaynet.pth"
    
    # Check if weights already exist and are not empty
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 1000000:  # > 1MB
        print(f"✅ Weights already exist at {WEIGHTS_PATH}")
        print(f"   Size: {WEIGHTS_PATH.stat().st_size / (1024**3):.2f} GB")
        return True
    
    print(f"📥 Downloading model weights from Google Drive...")
    print(f"   File ID: {DRIVE_FILE_ID}")
    print(f"   Destination: {WEIGHTS_PATH}")
    
    try:
        # Download using gdown
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, str(WEIGHTS_PATH), quiet=False)
        
        # Verify download
        if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 1000000:
            print(f"✅ Download successful!")
            print(f"   Size: {WEIGHTS_PATH.stat().st_size / (1024**3):.2f} GB")
            return True
        else:
            print(f"❌ Download failed or file is too small")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        return False


if __name__ == "__main__":
    success = download_weights()
    sys.exit(0 if success else 1)
