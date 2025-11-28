#!/usr/bin/env python3
"""
Download model weights from Google Drive
This script is used during Docker build or startup to download required weights
Includes timeout to prevent hanging on HuggingFace Spaces
"""
import os
import sys
import signal
from pathlib import Path
import gdown

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Download timed out")

def download_weights(timeout_seconds=300):
    """Download weights from Google Drive with timeout"""
    
    # Set timeout (5 minutes max for HuggingFace Spaces)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Google Drive file ID from the shared link
        # https://drive.google.com/file/d/1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz/view?usp=sharing
        DRIVE_FILE_ID = "1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz"
        
        # Destination path - works in Docker (/app) and local dev
        if os.path.exists("/app"):
            REPO_ROOT = Path("/app")
        else:
            REPO_ROOT = Path(__file__).parent.parent.parent.parent
        
        WEIGHTS_DIR = REPO_ROOT / "finetuning_rodla/finetuning_rodla/checkpoints"
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        
        WEIGHTS_PATH = WEIGHTS_DIR / "rodla_internimage_xl_publaynet.pth"
        
        # Check if weights already exist and are not empty
        if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 1000000:  # > 1MB
            print(f"✅ Weights already exist at {WEIGHTS_PATH}")
            print(f"   Size: {WEIGHTS_PATH.stat().st_size / (1024**3):.2f} GB")
            signal.alarm(0)  # Cancel alarm
            return True
        
        print(f"📥 Downloading model weights from Google Drive...")
        print(f"   File ID: {DRIVE_FILE_ID}")
        print(f"   Destination: {WEIGHTS_PATH}")
        print(f"   Timeout: {timeout_seconds} seconds")
        
        try:
            # Download using gdown
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, str(WEIGHTS_PATH), quiet=False)
            
            # Verify download
            if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 1000000:
                print(f"✅ Download successful!")
                print(f"   Size: {WEIGHTS_PATH.stat().st_size / (1024**3):.2f} GB")
                signal.alarm(0)  # Cancel alarm
                return True
            else:
                print(f"❌ Download failed or file is too small")
                signal.alarm(0)  # Cancel alarm
                return False
                
        except TimeoutError:
            print(f"⏱️  Download timed out after {timeout_seconds} seconds")
            print(f"   This is expected on HuggingFace Spaces. Backend will use heuristic detection.")
            signal.alarm(0)  # Cancel alarm
            return False
        except Exception as e:
            print(f"❌ Error downloading weights: {e}")
            signal.alarm(0)  # Cancel alarm
            return False
    
    except TimeoutError:
        print(f"⏱️  Download timed out after {timeout_seconds} seconds")
        print(f"   This is expected on HuggingFace Spaces. Backend will use heuristic detection.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        signal.alarm(0)  # Make sure alarm is cancelled


if __name__ == "__main__":
    # Allow timeout to be passed as argument, default to 5 minutes
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    success = download_weights(timeout_seconds=timeout)
    sys.exit(0 if success else 1)
