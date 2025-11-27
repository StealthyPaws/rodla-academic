"""Configuration and constants for the RoDLA API"""
from pathlib import Path
import sys

# Repository paths
REPO_ROOT = Path("/home/admin/CV/rodla-academic")
MODEL_CONFIG_PATH = REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"
MODEL_WEIGHTS_PATH = REPO_ROOT / "finetuning_rodla/finetuning_rodla/checkpoints/rodla_internimage_xl_publaynet.pth"

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


# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Job storage directory
JOBS_DIR = OUTPUT_DIR / "jobs"

# Batch processing limits
MAX_BATCH_SIZE = 300  # Maximum images per batch
MIN_BATCH_SIZE = 1    # Minimum images per batch

# Job retention
JOB_RETENTION_HOURS = 48  # Keep job metadata for 48 hours
MAX_CONCURRENT_JOBS = 10  # Maximum active jobs (future use)

# Batch output directory prefix
BATCH_OUTPUT_PREFIX = "batch_"

# Visualization modes
VISUALIZATION_MODES = ["none", "per_image", "summary", "both"]

# Default batch settings
DEFAULT_BATCH_CONFIG = {
    "score_threshold": 0.3,
    "visualization_mode": "none",
    "save_json": True,
    "perturbation_mode": "shared"
}

# Processing time estimation (seconds per image)
ESTIMATED_TIME_PER_IMAGE = 3.0
ESTIMATED_TIME_PER_VIZ = 0.5  # Additional time for visualizations

# Create jobs directory
JOBS_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Batch processing directories initialized:")
print(f"   - Jobs: {JOBS_DIR}")
print(f"   - Max batch size: {MAX_BATCH_SIZE} images")