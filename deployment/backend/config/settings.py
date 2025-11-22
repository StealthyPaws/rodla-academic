"""Configuration and constants for the RoDLA API"""
from pathlib import Path
import sys

# Repository paths
REPO_ROOT = Path("/mnt/d/MyStuff/University/Current/CV/Project/RoDLA")
MODEL_CONFIG_PATH = REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"
MODEL_WEIGHTS_PATH = REPO_ROOT / "rodla_internimage_xl_m6doc.pth"

# Add to Python path
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "model"))
sys.path.append(str(REPO_ROOT / "model/ops_dcnv3"))

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration
DEFAULT_SCORE_THRESHOLD = 0.3
MAX_DETECTIONS_PER_IMAGE = 300

# Baseline performance for mRD calculation (from paper)
BASELINE_MAP = {
    "M6Doc": 70.0,
    "PubLayNet": 96.0,
    "DocLayNet": 80.5
}

# Model information
MODEL_INFO = {
    "model_name": "RoDLA InternImage-XL",
    "paper": "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models (CVPR 2024)",
    "backbone": "InternImage-XL",
    "detection_framework": "DINO with Channel Attention + Average Pooling",
    "dataset": "M6Doc-P",
    "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
    "state_of_the_art_performance": {
        "clean_mAP": 70.0,
        "perturbed_avg_mAP": 61.7,
        "mRD_score": 147.6
    }
}

# API settings
API_TITLE = "RoDLA Object Detection API"
API_HOST = "0.0.0.0"
API_PORT = 8000

# CORS settings
CORS_ORIGINS = ["*"]
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# ============================================================================
# PERTURBATION CONFIGURATION
# ============================================================================

# Perturbation output directory
PERTURBATION_OUTPUT_DIR = OUTPUT_DIR / "perturbations"

# Maximum number of perturbations per request
MAX_PERTURBATIONS_PER_REQUEST = 5

# Background images folder (default location)
DEFAULT_BACKGROUND_FOLDER = REPO_ROOT / "perturbation" / "background_image"