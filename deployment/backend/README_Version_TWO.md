# RoDLA Document Layout Analysis API

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![CVPR](https://img.shields.io/badge/CVPR-2024-purple.svg)

**A Production-Ready API for Robust Document Layout Analysis with Perturbation Testing**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Perturbations](#-perturbation-system) • [Architecture](#-architecture)

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [System Requirements](#-system-requirements)
4. [Installation](#-installation)
5. [Quick Start](#-quick-start)
6. [Project Structure](#-project-structure)
7. [Architecture Deep Dive](#-architecture-deep-dive)
8. [Configuration](#-configuration)
9. [API Reference](#-api-reference)
10. [Perturbation System](#-perturbation-system)
11. [Metrics System](#-metrics-system)
12. [Visualization Engine](#-visualization-engine)
13. [Services Layer](#-services-layer)
14. [Utilities Reference](#-utilities-reference)
15. [Error Handling](#-error-handling)
16. [Performance Optimization](#-performance-optimization)
17. [Security Considerations](#-security-considerations)
18. [Testing](#-testing)
19. [Deployment](#-deployment)
20. [Troubleshooting](#-troubleshooting)
21. [Contributing](#-contributing)
22. [Citation](#-citation)
23. [License](#-license)

---

## 🎯 Overview

### What is RoDLA?

RoDLA (Robust Document Layout Analysis) is a state-of-the-art deep learning model for detecting and classifying layout elements in document images. Published at **CVPR 2024**, it focuses on robustness to various perturbations including noise, blur, and geometric distortions.

The model achieves:
- **70.0% mAP** on clean M6Doc dataset
- **61.7% average mAP** on perturbed images
- **147.6 mRD** (Mean Robustness Degradation) score
- Detection of **74 document element classes**

### What is this API?

This repository provides a **production-ready FastAPI wrapper** around the RoDLA model, featuring:

#### Core Detection Features
- 🔍 RESTful API endpoints for document analysis
- 📊 Comprehensive metrics calculation (20+ metrics)
- 📈 Automated visualization generation (8 chart types)
- 🛡️ Robustness assessment based on the RoDLA paper
- 🧠 Human-readable interpretation of results
- 📁 Flexible output formats (JSON, annotated images)

#### NEW: Perturbation Testing Features
- 🎨 **12 perturbation types** across 5 categories
- 🔬 **Apply-only mode** - Test perturbations without detection
- 🎯 **Combined mode** - Perturb then detect in one request
- 📊 **Perturbation analytics** - Track success rates and effects
- 💾 **Save perturbed images** - Keep transformed images for analysis
- 🔄 **Sequential application** - Apply multiple perturbations in order

### Key Statistics

| Metric | Value |
|--------|-------|
| Clean mAP (M6Doc) | 70.0% |
| Perturbed Average mAP | 61.7% |
| mRD Score | 147.6 |
| Max Detections/Image | 300 |
| Supported Classes | 74 (M6Doc) |
| **Perturbation Types** | **12** |
| **Perturbation Categories** | **5** |
| **Intensity Levels** | **3 (mild/moderate/severe)** |

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🔍 **Multi-class Detection** | Detect 74+ document element types |
| 📊 **Comprehensive Metrics** | 20+ analytical metrics per image |
| 📈 **Auto Visualization** | 8 chart types generated automatically |
| 🛡️ **Robustness Analysis** | mPE and mRD estimation |
| 🧠 **Smart Interpretation** | Human-readable analysis summaries |
| ⚡ **GPU Acceleration** | CUDA support for fast inference |
| 📁 **Flexible Output** | JSON, annotated images, or both |

### NEW: Perturbation Capabilities

| Feature | Description |
|---------|-------------|
| 🎨 **12 Perturbation Types** | Blur, noise, content, inconsistency, spatial |
| 🔬 **Independent Testing** | Apply perturbations without detection |
| 🎯 **Integrated Pipeline** | Perturb + detect in single request |
| 📊 **Success Tracking** | Monitor which perturbations succeed/fail |
| 💾 **Save Outputs** | Persist perturbed images to disk |
| 🔄 **Sequential Processing** | Apply multiple effects in order |
| 📈 **Robustness Testing** | Evaluate model performance under stress |

### Document Element Types

The model can detect various document elements including:

```
Text Elements:        Structural Elements:    Visual Elements:
├── Paragraph         ├── Header              ├── Figure
├── Title             ├── Footer              ├── Table
├── Caption           ├── Page Number         ├── Chart
├── List              ├── Section             ├── Logo
├── Footnote          ├── Column              ├── Stamp
└── Abstract          └── Margin              └── Equation
```

### Perturbation Categories

```
Blur Effects:         Noise Effects:          Content Effects:
├── Defocus           ├── Speckle             ├── Watermark
└── Vibration         └── Texture             └── Background

Inconsistency:        Spatial Transform:
├── Ink Holdout       ├── Rotation
├── Ink Bleeding      ├── Keystoning
└── Illumination      └── Warping
```

---

## 💻 System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| GPU | 8 GB VRAM | 16+ GB VRAM |
| Storage | 10 GB | 20 GB |

### Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.8 - 3.10 |
| CUDA | 11.7+ |
| cuDNN | 8.5+ |
| OS | Linux (Ubuntu 20.04+) / WSL2 |

### Python Dependencies

```python
# Core Framework
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# ML/Deep Learning
torch>=2.0.0
mmdet>=3.0.0
mmcv>=2.0.0
detectron2>=0.6

# Data Processing
numpy>=1.24.0
pillow>=9.5.0
opencv-python>=4.8.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Perturbation Dependencies (NEW)
ocrodeg>=0.3.0
imgaug>=0.4.0
pyiqa>=0.1.7

# Utilities
pydantic>=2.0.0
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rodla-api.git
cd rodla-api/deployment
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n rodla python=3.9
conda activate rodla

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: Install MMDetection

```bash
pip install -U openmim
mim install mmengine
mim install mmcv>=2.0.0
mim install mmdet>=3.0.0
```

### Step 5: Install Detectron2

```bash
# Install pre-built version
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Or build from source
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

### Step 6: Install Perturbation Dependencies

```bash
# Install perturbation-specific libraries
pip install ocrodeg>=0.3.0
pip install imgaug>=0.4.0
pip install pyiqa>=0.1.7

# Verify installation
python -c "import ocrodeg, imgaug; print('Perturbation dependencies OK')"
```

### Step 7: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 8: Download Model Weights

```bash
# Create weights directory
mkdir -p weights

# Download from official source
wget https://path-to-weights/rodla_internimage_xl_m6doc.pth \
  -O weights/rodla_internimage_xl_m6doc.pth

# Verify file integrity
ls -lh weights/rodla_internimage_xl_m6doc.pth
```

### Step 9: Copy Perturbation Modules

```bash
# Copy perturbation files from RoDLA root to deployment
cd /path/to/RoDLA
cp perturbation/blur.py deployment/perturbations/
cp perturbation/content.py deployment/perturbations/
cp perturbation/noise.py deployment/perturbations/
cp perturbation/inconsistency.py deployment/perturbations/
cp perturbation/spatial.py deployment/perturbations/

# Verify files are copied
ls deployment/perturbations/
```

### Step 10: Setup Background Images (Optional)

```bash
# For background perturbation, setup background images
mkdir -p perturbation/background_image

# Copy or download background textures
cp /path/to/backgrounds/*.jpg perturbation/background_image/

# Create background file index
cd perturbation/background_image
python -c "from content import create_background_file; create_background_file('.')"
```

### Step 11: Configure Paths

Edit `config/settings.py`:

```python
from pathlib import Path

# Update these paths to match your setup
REPO_ROOT = Path("/path/to/your/RoDLA")
MODEL_CONFIG = REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"
MODEL_WEIGHTS = REPO_ROOT / "rodla_internimage_xl_m6doc.pth"

# Output directories (created automatically)
OUTPUT_DIR = Path("outputs")
PERTURBATION_OUTPUT_DIR = OUTPUT_DIR / "perturbations"

# Background folder for perturbations
DEFAULT_BACKGROUND_FOLDER = REPO_ROOT / "perturbation" / "background_image"
```

### Step 12: Verify Installation

```bash
# Test imports
python -c "
import torch
import mmdet
import detectron2
import ocrodeg
import imgaug
from fastapi import FastAPI
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test API startup
python backend.py &
sleep 5
curl http://localhost:8000/api/model-info
```

---

## ⚡ Quick Start

### Starting the Server

```bash
# Development mode (with auto-reload)
python backend.py

# Production mode with uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 1

# With specific log level
uvicorn backend:app --host 0.0.0.0 --port 8000 --log-level info
```

**Expected Output:**
```
============================================================
Starting RoDLA Document Layout Analysis API
============================================================
📁 Creating output directories...
   ✓ Main output: outputs
   ✓ Perturbations: outputs/perturbations

🔧 Loading RoDLA model...
   Loading checkpoint...
   ✓ Model loaded successfully

============================================================
✅ API Ready!
============================================================
🌐 Main API: http://0.0.0.0:8000
📚 Docs: http://0.0.0.0:8000/docs
📖 ReDoc: http://0.0.0.0:8000/redoc

🎯 Available Endpoints:
   • GET  /api/model-info              - Model information
   • POST /api/detect                  - Standard detection
   • GET  /api/perturbations/info      - Perturbation info (NEW)
   • POST /api/perturb                 - Apply perturbations (NEW)
   • POST /api/detect-with-perturbation - Detect with perturbations (NEW)
============================================================
```

### Making Your First Request

#### 1. Get Model Information

```bash
curl http://localhost:8000/api/model-info
```

#### 2. Basic Detection

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -H "accept: application/json" \
  -F "file=@document.jpg" \
  -F "score_thr=0.3" \
  -F "generate_visualizations=true"
```

#### 3. Get Perturbation Information (NEW)

```bash
curl http://localhost:8000/api/perturbations/info
```

#### 4. Apply Perturbation (NEW)

```bash
curl -X POST "http://localhost:8000/api/perturb" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"defocus","degree":2}]' \
  -F "save_image=true"
```

#### 5. Detect with Perturbation (NEW)

```bash
curl -X POST "http://localhost:8000/api/detect-with-perturbation" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"rotation","degree":1}]' \
  -F "score_thr=0.3"
```

### Python Client Examples

#### Basic Detection

```python
import requests

# Upload and analyze document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/detect",
        files={"file": f},
        data={
            "score_thr": "0.3",
            "return_image": "false",
            "generate_visualizations": "true"
        }
    )

result = response.json()
print(f"Detected {result['core_results']['summary']['total_detections']} elements")
print(f"Average confidence: {result['core_results']['summary']['average_confidence']:.2%}")
```

#### Apply Perturbations Only (NEW)

```python
import requests
import json

# Apply multiple perturbations
with open("document.jpg", "rb") as f:
    perturbations = [
        {"type": "defocus", "degree": 2},
        {"type": "speckle", "degree": 1},
        {"type": "rotation", "degree": 1}
    ]
    
    response = requests.post(
        "http://localhost:8000/api/perturb",
        files={"file": f},
        data={
            "perturbations": json.dumps(perturbations),
            "save_image": "true",
            "return_base64": "false"
        }
    )
    
    result = response.json()
    print(f"Success: {result['success']}")
    print(f"Applied: {len(result['perturbations_applied'])}")
    print(f"Failed: {len(result['perturbations_failed'])}")
    print(f"Success rate: {result['success_rate']:.1%}")
    
    if result['saved_path']:
        print(f"Saved to: {result['saved_path']}")
```

#### Detect with Perturbations (NEW)

```python
import requests
import json

# Perturb then detect
with open("document.jpg", "rb") as f:
    perturbations = [
        {"type": "illumination", "degree": 2},
        {"type": "texture", "degree": 1}
    ]
    
    response = requests.post(
        "http://localhost:8000/api/detect-with-perturbation",
        files={"file": f},
        data={
            "perturbations": json.dumps(perturbations),
            "score_thr": "0.3",
            "save_perturbed_image": "true",
            "generate_visualizations": "true"
        }
    )
    
    result = response.json()
    
    # Detection results
    print(f"Detections: {result['core_results']['summary']['total_detections']}")
    print(f"Confidence: {result['core_results']['summary']['average_confidence']:.2%}")
    
    # Perturbation results
    if 'perturbation_info' in result:
        print(f"\nPerturbations applied: {len(result['perturbation_info']['applied'])}")
        print(f"Success rate: {result['perturbation_info']['success_rate']:.1%}")
```

---

## 📁 Project Structure

```
deployment/
├── backend.py                    # 🚀 Main FastAPI application entry point
├── requirements.txt              # 📦 Python dependencies
├── README.md                     # 📖 This comprehensive documentation
├── test_perturbations.html       # 🧪 Interactive web-based testing interface
│
├── config/                       # ⚙️ Configuration Layer
│   ├── __init__.py              #    Package initializer
│   └── settings.py              #    All configuration constants (UPDATED)
│
├── core/                         # 🧠 Core Application Layer
│   ├── __init__.py              #    Package initializer
│   ├── model_loader.py          #    Singleton model management
│   └── dependencies.py          #    FastAPI dependency injection
│
├── api/                          # 🌐 API Layer
│   ├── __init__.py              #    Package initializer
│   ├── routes.py                #    API endpoint definitions (UPDATED)
│   └── schemas.py               #    Pydantic request/response models (UPDATED)
│
├── services/                     # 🔧 Business Logic Layer
│   ├── __init__.py              #    Package initializer
│   ├── detection.py             #    Core detection logic
│   ├── processing.py            #    Result aggregation
│   ├── visualization.py         #    Chart generation (350+ lines)
│   ├── interpretation.py        #    Human-readable insights
│   └── perturbation.py          # 🆕 Perturbation service (NEW)
│
├── perturbations/                # 🎨 Perturbation Module (NEW)
│   ├── __init__.py              #    Exports all perturbation functions
│   ├── apply.py                 #    Main orchestration logic
│   ├── blur.py                  #    Defocus, vibration effects
│   ├── content.py               #    Watermark, background additions
│   ├── noise.py                 #    Speckle, texture noise
│   ├── inconsistency.py         #    Ink effects, illumination
│   └── spatial.py               #    Rotation, keystoning, warping
│
├── utils/                        # 🛠️ Utility Layer
│   ├── __init__.py              #    Package initializer
│   ├── helpers.py               #    General helper functions
│   ├── serialization.py         #    JSON conversion utilities
│   └── metrics/                 #    Metrics calculation modules
│       ├── __init__.py          #    Metrics package initializer
│       ├── core.py              #    Core detection metrics
│       ├── rodla.py             #    RoDLA-specific metrics
│       ├── spatial.py           #    Spatial distribution analysis
│       └── quality.py           #    Quality & complexity metrics
│
└── outputs/                      # 📤 Output Directory (auto-created)
    ├── *.json                   #    Detection results
    ├── *.png                    #    Visualization images
    └── perturbations/           # 🆕 Perturbed images (NEW)
        └── *.png                #    Saved perturbed images
```

### File Count Summary

| Layer | Files | Purpose | Status |
|-------|-------|---------|--------|
| Config | 2 | Configuration management | Updated |
| Core | 3 | Model and dependency management | Stable |
| API | 3 | HTTP endpoints and schemas | Updated |
| Services | 6 | Business logic implementation | +1 New |
| **Perturbations** | **7** | **Perturbation operations** | **NEW** |
| Utils | 7 | Helper functions and metrics | Stable |
| **Total** | **28** | **Complete modular architecture** | **+7 files** |

---

## 🏗️ Architecture Deep Dive

### Layered Architecture with Perturbations

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                               │
│              (Web Browser / API Clients)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP Requests
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                 │
│                    api/routes.py                                │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │GET /model-   │ │POST /api/    │ │GET /api/perturbations/  │ │
│  │    info      │ │     detect   │ │    info (NEW)           │ │
│  └──────────────┘ └──────────────┘ └─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┤
│  │POST /api/perturb (NEW)         POST /api/detect-with-      │ │
│  │                                      perturbation (NEW)     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Validated Requests
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICES LAYER                               │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────────────┐  │
│  │detection.py  │ │processing.py │ │visualization.py        │  │
│  │• Inference   │ │• Aggregate   │ │• 8 Chart Types         │  │
│  │• Processing  │ │• Save JSON   │ │• Base64 Encoding       │  │
│  └──────────────┘ └──────────────┘ └────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            interpretation.py                              │  │
│  │         • Human-readable insights                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  🆕 perturbation.py (NEW)                                 │  │
│  │  • Apply perturbations  • Save perturbed images           │  │
│  │  • Track success/failure • Generate metadata              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Data Processing
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UTILITIES LAYER                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  utils/metrics/                            │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐      │ │
│  │  │core.py  │ │rodla.py │ │spatial. │ │quality.py   │      │ │
│  │  │         │ │         │ │  py     │ │             │      │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘      │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌──────────────┐ ┌────────────────────────────────────┐       │
│  │helpers.py    │ │serialization.py                    │       │
│  └──────────────┘ └────────────────────────────────────┘       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Perturbation Operations
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  🆕 PERTURBATIONS LAYER (NEW)                                   │
│                 perturbations/                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    apply.py                                 │ │
│  │     • Orchestrates all perturbation types                  │ │
│  │     • Sequential application logic                         │ │
│  │     • Error handling and validation                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │blur.py   │ │noise.py  │ │content.  │ │inconsistency.py    │ │
│  │• Defocus │ │• Speckle │ │  py      │ │• Ink holdout       │ │
│  │•Vibration│ │• Texture │ │•Watermark│ │• Ink bleeding      │ │
│  │          │ │          │ │•Backgrnd │ │• Illumination      │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────│
│  │               spatial.py                                    │ │
│  │     • Rotation  • Keystoning  • Warping                     │ │
│  └──────────────────────────────────────────────────────────────│
└─────────────────────────┬───────────────────────────────────────┘
                          │ Model Operations
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE LAYER                                 │
│  ┌─────────────────────────┐ ┌───────────────────────────┐     │
│  │    model_loader.py      │ │    dependencies.py        │     │
│  │                         │ │                           │     │
│  │ • Singleton Pattern     │ │ • FastAPI DI              │     │
│  │ • GPU Management        │ │ • Model Injection         │     │
│  │ • Lazy Loading          │ │                           │     │
│  └─────────────────────────┘ └───────────────────────────┘     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Configuration
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIG LAYER                                 │
│                   config/settings.py (UPDATED)                  │
│  • Paths  • Constants  • Baseline Metrics  • Thresholds        │
│  🆕 • Perturbation Config  • Background Folders  • Categories   │
└─────────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Singleton** | `model_loader.py` | Single model instance for efficiency |
| **Factory** | `visualization.py` | Create multiple chart types dynamically |
| **Dependency Injection** | `dependencies.py` | Inject model into routes cleanly |
| **Repository** | `processing.py` | Abstract data persistence layer |
| **Facade** | `routes.py` | Simplify complex subsystem interactions |
| **Strategy** | `metrics/`, `perturbations/` | Interchangeable algorithms |
| **🆕 Chain of Responsibility** | `perturbations/apply.py` | Sequential perturbation application |
| **🆕 Template Method** | `perturbations/*.py` | Common perturbation interface |

### Data Flow Diagram - Standard Detection

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Image   │───▶│  Upload  │───▶│  Temp    │───▶│  Model   │
│  File    │    │  Handler │    │  File    │    │ Inference│
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                      │
      ┌───────────────────────────────────────────────┘
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Generate │───▶│ Assemble │───▶│  JSON    │
│ Interp.  │    │ Response │    │ Response │
└──────────┘    └──────────┘    └──────────┘
```

### Data Flow Diagram - Perturbation Pipeline (NEW)

```
                    PERTURBATION-ONLY MODE
┌──────────┐    ┌──────────┐    ┌──────────────────────┐
│  Image   │───▶│  Upload  │───▶│ Perturbation Config  │
│  File    │    │  Handler │    │  (Type + Degree)     │
└──────────┘    └──────────┘    └──────────┬───────────┘
                                            │
      ┌─────────────────────────────────────┘
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Apply Pert 1 │───▶│ Apply Pert 2 │───▶│ Apply Pert N │
│  (Defocus)   │    │  (Speckle)   │    │  (Rotation)  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
      ┌────────────────────────────────────────┘
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Track        │───▶│ Save Image   │───▶│   Return     │
│ Success/Fail │    │  (Optional)  │    │   Results    │
└──────────────┘    └──────────────┘    └──────────────┘

                 PERTURBATION + DETECTION MODE
┌──────────┐    ┌──────────┐    ┌──────────────────────┐
│  Image   │───▶│  Upload  │───▶│ Perturbation Config  │
│  File    │    │  Handler │    │  + Detection Config  │
└──────────┘    └──────────┘    └──────────┬───────────┘
                                            │
      ┌─────────────────────────────────────┘
      ▼
┌──────────────────┐    ┌──────────────┐    ┌──────────────┐
│ Apply            │───▶│ Save         │───▶│ Run Model    │
│ Perturbations    │    │ Temp Image   │    │ Inference    │
└──────────────────┘    └──────────────┘    └──────┬───────┘
                                                    │
      ┌─────────────────────────────────────────────┘
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Process      │───▶│ Calculate    │───▶│ Add Pert     │
│ Detections   │    │ All Metrics  │    │ Metadata     │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
      ┌────────────────────────────────────────┘
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Generate     │───▶│ Assemble     │───▶│ Return Full  │
│ Visualize    │    │ Response     │    │ Results      │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## ⚙️ Configuration

### config/settings.py (UPDATED)

This file centralizes all configuration parameters including new perturbation settings.

```python
"""
Configuration Settings Module
=============================
All application constants and configuration in one place.
Now includes perturbation configuration!
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Root directory of the RoDLA model repository
REPO_ROOT = Path("/mnt/d/MyStuff/University/Current/CV/Project/RoDLA")

# Model configuration file path
MODEL_CONFIG = REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"

# Pre-trained model weights path
MODEL_WEIGHTS = REPO_ROOT / "rodla_internimage_xl_m6doc.pth"

# Output directory for results and visualizations
OUTPUT_DIR = Path("outputs")

# 🆕 NEW: Perturbation output directory
PERTURBATION_OUTPUT_DIR = OUTPUT_DIR / "perturbations"

# =============================================================================
# API CONFIGURATION
# =============================================================================

# CORS settings
CORS_ORIGINS = ["*"]  # Restrict in production
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# API metadata
API_TITLE = "RoDLA Object Detection API"
API_VERSION = "2.1.0"  # 🆕 Bumped for perturbation feature
API_DESCRIPTION = "Production-ready API for Robust Document Layout Analysis with Perturbation Testing"
API_HOST = "0.0.0.0"
API_PORT = 8000

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default confidence threshold for detections
DEFAULT_SCORE_THRESHOLD = 0.3

# Maximum number of detections per image
MAX_DETECTIONS = 300

# Model metadata
MODEL_INFO = {
    "name": "RoDLA InternImage-XL",
    "paper": "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models",
    "conference": "CVPR 2024",
    "backbone": "InternImage-XL",
    "framework": "DINO with Channel Attention + Average Pooling",
    "dataset": "M6Doc-P"
}

# =============================================================================
# 🆕 PERTURBATION CONFIGURATION (NEW)
# =============================================================================

# Maximum number of perturbations per request
MAX_PERTURBATIONS_PER_REQUEST = 5

# Default background images folder
DEFAULT_BACKGROUND_FOLDER = REPO_ROOT / "perturbation" / "background_image"

# Perturbation categories and their types
PERTURBATION_CATEGORIES = {
    "blur": {
        "types": ["defocus", "vibration"],
        "description": "Blur effects simulating optical issues",
        "typical_use": "Test robustness to camera/scanning quality"
    },
    "noise": {
        "types": ["speckle", "texture"],
        "description": "Noise patterns and texture artifacts",
        "typical_use": "Test robustness to paper quality and printing defects"
    },
    "content": {
        "types": ["watermark", "background"],
        "description": "Content additions like watermarks and backgrounds",
        "typical_use": "Test robustness to document modifications"
    },
    "inconsistency": {
        "types": ["ink_holdout", "ink_bleeding", "illumination"],
        "description": "Print quality issues and lighting variations",
        "typical_use": "Test robustness to printing and lighting conditions"
    },
    "spatial": {
        "types": ["rotation", "keystoning", "warping"],
        "description": "Geometric transformations",
        "typical_use": "Test robustness to document positioning and distortion"
    }
}

# Flatten for validation
ALL_PERTURBATIONS = [
    p for category in PERTURBATION_CATEGORIES.values() 
    for p in category["types"]
]

# Degree intensity descriptions
PERTURBATION_DEGREES = {
    1: "Mild - Subtle effect, barely noticeable",
    2: "Moderate - Noticeable effect, moderate degradation", 
    3: "Severe - Strong effect, significant degradation"
}

# Perturbation-specific parameters
PERTURBATION_PARAMS = {
    "defocus": {
        "base_kernel_size": 1,
        "description": "Gaussian blur simulating out-of-focus camera"
    },
    "vibration": {
        "base_size": 3,
        "description": "Motion blur simulating camera shake"
    },
    "speckle": {
        "base_density": 1e-4,
        "description": "Random black/white spots"
    },
    "texture": {
        "base_fibers": 300,
        "description": "Paper texture and fiber patterns"
    },
    "watermark": {
        "default_text": "Watermark_test",
        "description": "Semi-transparent text overlay"
    },
    "background": {
        "requires_folder": True,
        "description": "Background image overlay"
    },
    "ink_holdout": {
        "description": "Ink not reaching paper (erosion effect)"
    },
    "ink_bleeding": {
        "description": "Ink spreading beyond intended areas"
    },
    "illumination": {
        "description": "Uneven lighting and shadows"
    },
    "rotation": {
        "max_angle_deg": [5, 10, 15],
        "description": "Document rotation"
    },
    "keystoning": {
        "max_perspective": [0.02, 0.04, 0.06],
        "description": "Perspective distortion"
    },
    "warping": {
        "description": "Elastic deformation"
    }
}

# =============================================================================
# BASELINE PERFORMANCE METRICS
# =============================================================================

# Clean performance baselines from the RoDLA paper (mAP scores)
BASELINE_MAP = {
    "M6Doc": 70.0,      # Main evaluation dataset
    "PubLayNet": 96.0,  # Scientific documents
    "DocLayNet": 80.5   # Diverse document types
}

# State-of-the-art performance metrics
SOTA_PERFORMANCE = {
    "clean_mAP": 70.0,
    "perturbed_avg_mAP": 61.7,
    "mRD_score": 147.6
}

# =============================================================================
# ANALYSIS THRESHOLDS
# =============================================================================

# Size distribution thresholds (as percentage of image area)
SIZE_THRESHOLDS = {
    "tiny": 0.005,      # < 0.5% of image
    "small": 0.02,      # 0.5% - 2%
    "medium": 0.1,      # 2% - 10%
    "large": 1.0        # >= 10%
}

# Confidence level thresholds
CONFIDENCE_THRESHOLDS = {
    "very_high": 0.9,
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4
}

# Robustness assessment thresholds
ROBUSTNESS_THRESHOLDS = {
    "mPE_low": 20,
    "mPE_medium": 40,
    "mRD_excellent": 100,
    "mRD_good": 150,
    "cv_stable": 0.15,
    "cv_moderate": 0.30
}

# Complexity scoring weights
COMPLEXITY_WEIGHTS = {
    "class_diversity": 30,
    "detection_count": 30,
    "density": 20,
    "clustering": 20
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

# Figure sizes for different chart types
FIGURE_SIZES = {
    "bar_chart": (12, 6),
    "histogram": (10, 6),
    "heatmap": (10, 8),
    "boxplot": (12, 6),
    "scatter": (10, 6),
    "pie": (8, 8)
}

# Color schemes
COLOR_SCHEMES = {
    "primary": "steelblue",
    "secondary": "forestgreen",
    "accent": "coral",
    "heatmap": "YlOrRd",
    "scatter": "viridis"
}

# DPI for saved images
VISUALIZATION_DPI = 100
```

### Environment Variables

For production deployments, use environment variables:

```bash
# .env file
RODLA_REPO_ROOT=/path/to/RoDLA
RODLA_MODEL_CONFIG=model/configs/m6doc/rodla_internimage_xl_m6doc.py
RODLA_MODEL_WEIGHTS=rodla_internimage_xl_m6doc.pth
RODLA_OUTPUT_DIR=outputs
RODLA_PERTURBATION_DIR=outputs/perturbations
RODLA_DEFAULT_THRESHOLD=0.3
RODLA_API_HOST=0.0.0.0
RODLA_API_PORT=8000
RODLA_BACKGROUND_FOLDER=/path/to/backgrounds  # NEW
RODLA_MAX_PERTURBATIONS=5                     # NEW
```

---

## 🌐 API Reference

### Endpoints Overview

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/api/model-info` | Get model metadata | Stable |
| POST | `/api/detect` | Analyze document image | Stable |
| **GET** | **`/api/perturbations/info`** | **Get perturbation information** | **🆕 NEW** |
| **POST** | **`/api/perturb`** | **Apply perturbations only** | **🆕 NEW** |
| **POST** | **`/api/detect-with-perturbation`** | **Perturb then detect** | **🆕 NEW** |
| GET | `/health` | Health check | Optional |
| GET | `/docs` | Swagger UI documentation | Auto-generated |
| GET | `/redoc` | ReDoc documentation | Auto-generated |

---

### GET /api/model-info

Returns comprehensive information about the loaded model.

#### Request

```http
GET /api/model-info HTTP/1.1
Host: localhost:8000
```

#### Response

```json
{
    "model_name": "RoDLA InternImage-XL",
    "paper": "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models (CVPR 2024)",
    "num_classes": 74,
    "classes": [
        "paragraph", "title", "figure", "table", "caption",
        "header", "footer", "page_number", "list", "abstract",
        // ... 64 more classes
    ],
    "backbone": "InternImage-XL",
    "detection_framework": "DINO with Channel Attention + Average Pooling",
    "dataset": "M6Doc-P",
    "max_detections_per_image": 300,
    "state_of_the_art_performance": {
        "clean_mAP": 70.0,
        "perturbed_avg_mAP": 61.7,
        "mRD_score": 147.6
    }
}
```

---

### POST /api/detect

Analyzes a document image and returns comprehensive detection results.

#### Request

```http
POST /api/detect HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file: <binary image data>
score_thr: "0.3"
return_image: "false"
save_json: "true"
generate_visualizations: "true"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Image file (JPEG, PNG, PDF, etc.) |
| `score_thr` | string | "0.3" | Confidence threshold (0.0-1.0) |
| `return_image` | string | "false" | Return annotated image instead of JSON |
| `save_json` | string | "true" | Save results to disk |
| `generate_visualizations` | string | "true" | Generate visualization charts |

#### Response Structure

The response includes:
- **Core results**: Detection summary and individual detections
- **RoDLA metrics**: Perturbation effect estimates
- **Spatial analysis**: Distribution patterns
- **Class analysis**: Per-class statistics
- **Confidence analysis**: Confidence distribution
- **Robustness indicators**: Stability metrics
- **Layout complexity**: Complexity assessment
- **Quality metrics**: Detection quality
- **Visualizations**: 8 chart types (base64)
- **Interpretation**: Human-readable insights

*(Full response JSON too large to include - see original documentation or API docs)*

---

### 🆕 GET /api/perturbations/info (NEW)

Returns comprehensive information about all available perturbation types.

#### Request

```http
GET /api/perturbations/info HTTP/1.1
Host: localhost:8000
```

#### Response

```json
{
    "total_perturbations": 12,
    "categories": {
        "blur": {
            "types": ["defocus", "vibration"],
            "description": "Blur effects simulating optical issues",
            "typical_use": "Test robustness to camera/scanning quality"
        },
        "noise": {
            "types": ["speckle", "texture"],
            "description": "Noise patterns and texture artifacts",
            "typical_use": "Test robustness to paper quality and printing defects"
        },
        "content": {
            "types": ["watermark", "background"],
            "description": "Content additions like watermarks and backgrounds",
            "typical_use": "Test robustness to document modifications"
        },
        "inconsistency": {
            "types": ["ink_holdout", "ink_bleeding", "illumination"],
            "description": "Print quality issues and lighting variations",
            "typical_use": "Test robustness to printing and lighting conditions"
        },
        "spatial": {
            "types": ["rotation", "keystoning", "warping"],
            "description": "Geometric transformations",
            "typical_use": "Test robustness to document positioning and distortion"
        }
    },
    "all_types": [
        "defocus", "vibration", "speckle", "texture",
        "watermark", "background", "ink_holdout", "ink_bleeding",
        "illumination", "rotation", "keystoning", "warping"
    ],
    "degree_levels": {
        "1": "Mild - Subtle effect, barely noticeable",
        "2": "Moderate - Noticeable effect, moderate degradation",
        "3": "Severe - Strong effect, significant degradation"
    },
    "notes": {
        "background": "Requires background_folder parameter",
        "spatial": "May change image dimensions slightly",
        "sequential": "Multiple perturbations applied in order specified",
        "max_per_request": "Maximum 5 perturbations per request"
    }
}
```

---

### 🆕 POST /api/perturb (NEW)

Apply perturbations to an image without performing detection. Perfect for testing perturbation effects or generating augmented datasets.

#### Request

```http
POST /api/perturb HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file: <binary image data>
perturbations: '[{"type":"defocus","degree":2},{"type":"speckle","degree":1}]'
return_base64: "false"
save_image: "true"
background_folder: "/path/to/backgrounds" (optional)
```

#### Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `file` | File | - | ✅ Yes | Image file to perturb |
| `perturbations` | JSON string | - | ✅ Yes | Array of perturbation configs |
| `return_base64` | string | "false" | ❌ No | Return perturbed image as base64 |
| `save_image` | string | "false" | ❌ No | Save perturbed image to disk |
| `background_folder` | string | None | ❌ No | Path for 'background' perturbation |

#### Perturbation Config Format

```json
[
    {
        "type": "defocus",           // Required: perturbation type
        "degree": 2,                 // Required: 1 (mild), 2 (moderate), 3 (severe)
        "background_folder": "..."   // Optional: only for 'background' type
    },
    {
        "type": "speckle",
        "degree": 1
    }
]
```

#### Response

```json
{
    "success": true,
    "message": "Applied 3/3 perturbations successfully",
    "perturbations_applied": [
        {
            "order": 1,
            "type": "defocus",
            "degree": 2,
            "message": "Successfully applied defocus (degree 2)"
        },
        {
            "order": 2,
            "type": "speckle",
            "degree": 1,
            "message": "Successfully applied speckle (degree 1)"
        },
        {
            "order": 3,
            "type": "rotation",
            "degree": 1,
            "message": "Successfully applied rotation (degree 1)"
        }
    ],
    "perturbations_failed": [],
    "total_perturbations": 3,
    "success_rate": 1.0,
    "original_shape": [3508, 2480, 3],
    "perturbed_shape": [3508, 2480, 3],
    "image_base64": null,  // Only if return_base64=true
    "saved_path": "outputs/perturbations/document_pert_defocus2_speckle1_rotation1_20241115_143022.png"
}
```

#### Example Usage

```bash
# Single perturbation
curl -X POST "http://localhost:8000/api/perturb" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"defocus","degree":2}]' \
  -F "save_image=true"

# Multiple perturbations with base64 return
curl -X POST "http://localhost:8000/api/perturb" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"defocus","degree":2},{"type":"speckle","degree":1},{"type":"rotation","degree":1}]' \
  -F "return_base64=true" \
  -F "save_image=true"

# Background perturbation (requires folder)
curl -X POST "http://localhost:8000/api/perturb" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"background","degree":2}]' \
  -F "background_folder=/path/to/backgrounds"
```

---

### 🆕 POST /api/detect-with-perturbation (NEW)

Apply perturbations to an image, then perform RoDLA detection on the perturbed result. This endpoint combines both operations in a single request.

#### Request

```http
POST /api/detect-with-perturbation HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data

file: <binary image data>
perturbations: '[{"type":"rotation","degree":1}]'
score_thr: "0.3"
return_image: "false"
save_json: "true"
generate_visualizations: "true"
save_perturbed_image: "false"
background_folder: "/path/to/backgrounds" (optional)
```

#### Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `file` | File | - | ✅ Yes | Image file to process |
| `perturbations` | JSON string | None | ❌ No | Array of perturbation configs |
| `score_thr` | string | "0.3" | ❌ No | Confidence threshold |
| `return_image` | string | "false" | ❌ No | Return annotated image |
| `save_json` | string | "true" | ❌ No | Save results to disk |
| `generate_visualizations` | string | "true" | ❌ No | Generate charts |
| `save_perturbed_image` | string | "false" | ❌ No | Save perturbed image |
| `background_folder` | string | None | ❌ No | Path for 'background' perturbation |

#### Response

Standard detection response PLUS perturbation metadata:

```json
{
    "success": true,
    "timestamp": "2024-01-15T10:30:45.123456",
    "filename": "document.jpg",
    
    // Standard detection fields...
    "image_info": {...},
    "detection_config": {...},
    "core_results": {...},
    "rodla_metrics": {...},
    "spatial_analysis": {...},
    // ... all other detection fields ...
    
    // 🆕 NEW: Perturbation information
    "perturbation_info": {
        "applied": [
            {
                "order": 1,
                "type": "rotation",
                "degree": 1,
                "message": "Successfully applied rotation (degree 1)"
            },
            {
                "order": 2,
                "type": "illumination",
                "degree": 2,
                "message": "Successfully applied illumination (degree 2)"
            }
        ],
        "failed": [],
        "success_rate": 1.0
    }
}
```

#### Example Usage

```bash
# Detect with single perturbation
curl -X POST "http://localhost:8000/api/detect-with-perturbation" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"defocus","degree":2}]' \
  -F "score_thr=0.3"

# Detect with multiple perturbations
curl -X POST "http://localhost:8000/api/detect-with-perturbation" \
  -F "file=@document.jpg" \
  -F 'perturbations=[{"type":"illumination","degree":2},{"type":"speckle","degree":1}]' \
  -F "score_thr=0.3" \
  -F "save_perturbed_image=true"

# Detect without perturbations (same as /api/detect)
curl -X POST "http://localhost:8000/api/detect-with-perturbation" \
  -F "file=@document.jpg" \
  -F "score_thr=0.3"
```

---

## 🎨 Perturbation System

### Overview

The perturbation system allows you to apply various image transformations that simulate real-world document degradation and variations. This is crucial for:

- **Robustness Testing**: Evaluate how well the model performs under adverse conditions
- **Data Augmentation**: Generate training data variations
- **Benchmark Creation**: Create standardized test sets
- **Research**: Study model behavior under controlled perturbations

### Perturbation Categories

#### 1. Blur Effects

Simulates optical issues from cameras or scanners.

| Type | Description | Use Case | Degree Effects |
|------|-------------|----------|----------------|
| **defocus** | Gaussian blur | Out-of-focus camera | 1: σ=1, 2: σ=3, 3: σ=5 |
| **vibration** | Motion blur | Camera shake | 1: 3px, 2: 9px, 3: 15px |

**Example:**
```json
[
    {"type": "defocus", "degree": 2},
    {"type": "vibration", "degree": 1}
]
```

#### 2. Noise Effects

Simulates paper and printing quality issues.

| Type | Description | Use Case | Degree Effects |
|------|-------------|----------|----------------|
| **speckle** | Random black/white spots | Paper defects | 1-3: Increasing density |
| **texture** | Fibrous paper texture | Paper grain | 1: 300 fibers, 3: 900 fibers |

**Example:**
```json
[
    {"type": "speckle", "degree": 1},
    {"type": "texture", "degree": 2}
]
```

#### 3. Content Effects

Adds overlays and modifications.

| Type | Description | Use Case | Degree Effects |
|------|-------------|----------|----------------|
| **watermark** | Semi-transparent text | Document marking | 1-3: Increasing opacity |
| **background** | Background image overlay | Texture addition | 1-3: More background images |

**Example:**
```json
[
    {"type": "watermark", "degree": 2},
    {"type": "background", "degree": 1, "background_folder": "/path/to/backgrounds"}
]
```

**Note**: `background` requires `background_folder` parameter.

#### 4. Inconsistency Effects

Simulates printing and lighting issues.

| Type | Description | Use Case | Degree Effects |
|------|-------------|----------|----------------|
| **ink_holdout** | Ink not reaching paper | Print quality | 1-3: Increasing erosion |
| **ink_bleeding** | Ink spreading | Poor paper quality | 1-3: Increasing dilation |
| **illumination** | Uneven lighting | Shadow/highlights | 1-3: Stronger effect |

**Example:**
```json
[
    {"type": "ink_holdout", "degree": 1},
    {"type": "ink_bleeding", "degree": 2},
    {"type": "illumination", "degree": 1}
]
```

#### 5. Spatial Transformations

Geometric distortions and positioning issues.

| Type | Description | Use Case | Degree Effects |
|------|-------------|----------|----------------|
| **rotation** | Document rotation | Scanning angle | 1: ±5°, 2: ±10°, 3: ±15° |
| **keystoning** | Perspective distortion | Camera angle | 1-3: Increasing distortion |
| **warping** | Elastic deformation | Page curl | 1-3: Increasing warping |

**Example:**
```json
[
    {"type": "rotation", "degree": 1},
    {"type": "keystoning", "degree": 2},
    {"type": "warping", "degree": 1}
]
```

**Note**: Spatial perturbations may modify bounding box annotations if provided.

### Intensity Levels

All perturbations support 3 intensity levels:

| Degree | Name | Description | Visual Impact |
|--------|------|-------------|---------------|
| **1** | Mild | Subtle, barely noticeable | Minimal degradation |
| **2** | Moderate | Noticeable, moderate effect | Visible but readable |
| **3** | Severe | Strong, significant effect | Clearly degraded |

### Sequential Application

Perturbations are applied in the order specified:

```json
[
    {"type": "defocus", "degree": 2},      // Applied first
    {"type": "speckle", "degree": 1},      // Applied to defocused image
    {"type": "rotation", "degree": 1}      // Applied to defocused+speckled image
]
```

**Important**: Order matters! Different orders produce different results.

### Perturbation Module Structure

```python
perturbations/
├── __init__.py              # Exports all functions
├── apply.py                 # Main orchestration
│   ├── apply_perturbation()        # Apply single perturbation
│   ├── apply_multiple_perturbations()  # Apply sequence
│   └── get_perturbation_info()     # Get available perturbations
├── blur.py                  # Defocus, vibration
├── content.py               # Watermark, background
├── noise.py                 # Speckle, texture
├── inconsistency.py         # Ink effects, illumination
└── spatial.py               # Rotation, keystoning, warping
```

### Usage Examples

#### Python Client

```python
import requests
import json

# Test different perturbation categories
perturbation_tests = {
    "blur_test": [
        {"type": "defocus", "degree": 2},
        {"type": "vibration", "degree": 1}
    ],
    "noise_test": [
        {"type": "speckle", "degree": 2},
        {"type": "texture", "degree": 1}
    ],
    "spatial_test": [
        {"type": "rotation", "degree": 1},
        {"type": "keystoning", "degree": 2}
    ],
    "combined_test": [
        {"type": "illumination", "degree": 2},
        {"type": "speckle", "degree": 1},
        {"type": "rotation", "degree": 1}
    ]
}

for test_name, perturbations in perturbation_tests.items():
    with open("document.jpg", "rb") as f:
        response = requests.post(
            "http://localhost:8000/api/perturb",
            files={"file": f},
            data={
                "perturbations": json.dumps(perturbations),
                "save_image": "true"
            }
        )
        
        result = response.json()
        print(f"\n{test_name}:")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Saved to: {result.get('saved_path', 'N/A')}")
```

#### Batch Processing

```python
import os
from pathlib import Path

def batch_perturb_documents(input_dir, perturbations):
    """Apply same perturbations to all images in directory."""
    input_path = Path(input_dir)
    results = []
    
    for image_file in input_path.glob("*.jpg"):
        with open(image_file, "rb") as f:
            response = requests.post(
                "http://localhost:8000/api/perturb",
                files={"file": f},
                data={
                    "perturbations": json.dumps(perturbations),
                    "save_image": "true"
                }
            )
            
            result = response.json()
            results.append({
                "filename": image_file.name,
                "success": result["success"],
                "success_rate": result["success_rate"]
            })
    
    return results

# Apply consistent perturbations to entire dataset
perturbations = [
    {"type": "defocus", "degree": 1},
    {"type": "speckle", "degree": 1}
]

results = batch_perturb_documents("input_images/", perturbations)
print(f"Processed {len(results)} images")
print(f"Overall success: {sum(r['success'] for r in results)}/{len(results)}")
```

#### Robustness Testing Pipeline

```python
def test_model_robustness(image_path, perturbation_types, degrees=[1, 2, 3]):
    """
    Test model performance across different perturbation intensities.
    """
    results = {}
    
    for pert_type in perturbation_types:
        results[pert_type] = {}
        
        for degree in degrees:
            with open(image_path, "rb") as f:
                response = requests.post(
                    "http://localhost:8000/api/detect-with-perturbation",
                    files={"file": f},
                    data={
                        "perturbations": json.dumps([{"type": pert_type, "degree": degree}]),
                        "score_thr": "0.3"
                    }
                )
                
                result = response.json()
                
                results[pert_type][f"degree_{degree}"] = {
                    "detections": result["core_results"]["summary"]["total_detections"],
                    "avg_confidence": result["core_results"]["summary"]["average_confidence"],
                    "robustness_score": result["robustness_indicators"]["robustness_rating"]["score"]
                }
    
    return results

# Test all blur perturbations at all intensities
robustness_results = test_model_robustness(
    "test_document.jpg",
    perturbation_types=["defocus", "vibration", "speckle"],
    degrees=[1, 2, 3]
)

# Analyze results
for pert_type, degrees in robustness_results.items():
    print(f"\n{pert_type}:")
    for degree, metrics in degrees.items():
        print(f"  {degree}: {metrics['detections']} detections, "
              f"{metrics['avg_confidence']:.2%} confidence")
```

---

## 📊 Metrics System

### Metrics Architecture

```
utils/metrics/
├── __init__.py          # Exports all metric functions
├── core.py              # Core detection metrics
├── rodla.py             # RoDLA-specific robustness metrics
├── spatial.py           # Spatial distribution analysis
└── quality.py           # Quality and complexity metrics
```

### Core Metrics (utils/metrics/core.py)

#### `calculate_core_metrics(detections, img_width, img_height)`

Computes fundamental detection statistics.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| `total_detections` | int | Number of detected elements | 0 - 300 |
| `unique_classes` | int | Number of distinct element types | 0 - 74 |
| `average_confidence` | float | Mean confidence score | 0.0 - 1.0 |
| `median_confidence` | float | Median confidence score | 0.0 - 1.0 |
| `min_confidence` | float | Lowest confidence | 0.0 - 1.0 |
| `max_confidence` | float | Highest confidence | 0.0 - 1.0 |
| `coverage_percentage` | float | % of image covered | 0.0 - 100.0 |
| `average_detection_area` | float | Mean area per detection | pixels² |

#### `calculate_class_metrics(detections)`

Per-class statistical analysis.

```python
{
    "paragraph": {
        "count": 15,
        "percentage": 31.91,
        "confidence_stats": {
            "mean": 0.8234,
            "std": 0.0876,
            "min": 0.6543,
            "max": 0.9654
        },
        "area_stats": {
            "mean": 125432.5,
            "std": 45678.2,
            "total": 1881487.5
        },
        "aspect_ratio_stats": {
            "mean": 2.345,
            "orientation": "horizontal"  # horizontal/vertical/square
        }
    },
    "title": {...},
    "figure": {...}
    // ... other classes
}
```

#### `calculate_confidence_metrics(detections)`

Detailed confidence distribution analysis.

**Confidence Bins:**
- **Very High**: 0.9 - 1.0 (highly certain)
- **High**: 0.8 - 0.9 (confident)
- **Medium**: 0.6 - 0.8 (acceptable)
- **Low**: 0.4 - 0.6 (uncertain)
- **Very Low**: 0.0 - 0.4 (very uncertain)

**Output:**
```python
{
    "distribution": {
        "mean": 0.7823,
        "median": 0.8156,
        "std": 0.1234,
        "min": 0.3012,
        "max": 0.9876,
        "q1": 0.7012,
        "q3": 0.8967
    },
    "binned_distribution": {
        "very_high": 15,
        "high": 20,
        "medium": 10,
        "low": 2,
        "very_low": 0
    },
    "percentages": {
        "very_high": 31.91,
        "high": 42.55,
        "medium": 21.28,
        "low": 4.26,
        "very_low": 0.0
    },
    "entropy": 2.3456  # Shannon entropy
}
```

---

### RoDLA Metrics (utils/metrics/rodla.py)

These metrics estimate robustness based on the RoDLA paper's methodology.

#### `calculate_rodla_metrics(detections, core_metrics)`

Estimates perturbation effects and robustness degradation.

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `estimated_mPE` | `(std × 100) + (range × 50)` | 0-100+ | Mean Perturbation Effect |
| `estimated_mRD` | `(degradation / mPE) × 100` | 0-200+ | Mean Robustness Degradation |
| `robustness_score` | `(1 - mRD/200) × 100` | 0-100 | Overall robustness |

**mPE Interpretation:**
```
low:    mPE < 20      → Minimal perturbation effect
                        Predictions are very consistent
                        
medium: 20 ≤ mPE < 40 → Moderate perturbation effect
                        Some variability in predictions
                        
high:   mPE ≥ 40      → Significant perturbation effect
                        High prediction variability
```

**mRD Interpretation:**
```
excellent:         mRD < 100     → Highly robust model
                                   Minimal performance degradation
                                   
good:              100 ≤ mRD < 150 → Acceptable robustness
                                    Moderate degradation
                                    
needs_improvement: mRD ≥ 150     → Robustness concerns
                                   Significant degradation
```

#### `calculate_robustness_indicators(detections, core_metrics)`

Stability and consistency metrics.

```python
{
    "stability_score": 87.65,        # (1 - CV) × 100
    "coefficient_of_variation": 0.12, # std / mean
    "high_confidence_ratio": 0.72,   # % detections with conf ≥ 0.8
    "prediction_consistency": "high", # Based on CV thresholds
    "model_certainty": "medium",      # Based on avg confidence
    "robustness_rating": {
        "rating": "good",             # excellent/good/fair/poor
        "score": 72.34                # Composite score
    }
}
```

**Robustness Rating Formula:**
```
score = (avg_confidence × 40) + 
        ((1 - CV) × 30) + 
        (high_conf_ratio × 30)

Rating categories:
- excellent: score ≥ 80  (very robust)
- good:      60 ≤ score < 80  (robust)
- fair:      40 ≤ score < 60  (acceptable)
- poor:      score < 40  (concerning)
```

---

### Spatial Metrics (utils/metrics/spatial.py)

#### `calculate_spatial_analysis(detections, img_width, img_height)`

Comprehensive spatial distribution analysis.

##### Horizontal Distribution
```python
{
    "mean": 1240.5,           # Mean x-coordinate
    "std": 456.7,             # Standard deviation
    "skewness": -0.234,       # Distribution asymmetry
                              # Negative = left-skewed
                              # Positive = right-skewed
    "left_third": 12,         # Count in left 33%
    "center_third": 25,       # Count in center 33%
    "right_third": 10         # Count in right 33%
}
```

##### Vertical Distribution
```python
{
    "mean": 1754.2,           # Mean y-coordinate
    "std": 892.4,             # Standard deviation
    "skewness": 0.156,        # Distribution asymmetry
                              # Negative = top-heavy
                              # Positive = bottom-heavy
    "top_third": 8,           # Count in top 33%
    "middle_third": 22,       # Count in middle 33%
    "bottom_third": 17        # Count in bottom 33%
}
```

##### Quadrant Distribution

Document divided into 4 equal quadrants:

```
┌─────────────┬─────────────┐
│     Q1      │     Q2      │
│  (top-left) │ (top-right) │
│             │             │
├─────────────┼─────────────┤
│     Q3      │     Q4      │
│ (bot-left)  │ (bot-right) │
│             │             │
└─────────────┴─────────────┘
```

```python
{
    "Q1": 12,  # Top-left
    "Q2": 15,  # Top-right
    "Q3": 10,  # Bottom-left
    "Q4": 10   # Bottom-right
}
```

##### Size Distribution

Elements categorized by area relative to image size:

| Category | Threshold | Description | Typical Elements |
|----------|-----------|-------------|------------------|
| **tiny** | < 0.5% | Very small | Footnotes, page numbers |
| **small** | 0.5% - 2% | Small | Captions, list items |
| **medium** | 2% - 10% | Medium | Paragraphs, titles |
| **large** | ≥ 10% | Large | Figures, tables |

```python
{
    "tiny": 5,
    "small": 15,
    "medium": 20,
    "large": 7
}
```

##### Density Metrics

```python
{
    "average_nearest_neighbor_distance": 234.56,  # pixels
    "spatial_clustering_score": 0.67              # 0-1 scale
                                                  # Higher = more clustered
                                                  # Lower = more dispersed
}
```

**Clustering Interpretation:**
- **0.0 - 0.3**: Highly dispersed (scattered layout)
- **0.3 - 0.7**: Moderately clustered (typical documents)
- **0.7 - 1.0**: Highly clustered (dense regions)

---

### Quality Metrics (utils/metrics/quality.py)

#### `calculate_layout_complexity(detections, img_width, img_height)`

Quantifies document structure complexity.

**Complexity Score Formula:**
```
score = (class_diversity / 20) × 30      # Max 20 unique classes
      + min(detections / 50, 1) × 30     # Detection count normalized
      + min(density / 10, 1) × 20        # Elements per megapixel
      + (1 - min(avg_dist / 500, 1)) × 20 # Spatial clustering
```

**Complexity Levels:**

| Level | Score Range | Description | Examples |
|-------|-------------|-------------|----------|
| **simple** | < 30 | Basic layout | Single-column text, minimal structure |
| **moderate** | 30 - 60 | Average complexity | Multi-column, some figures/tables |
| **complex** | ≥ 60 | Complex layout | Academic papers, magazines, forms |

**Layout Characteristics:**
```python
{
    "class_diversity": 12,              # Number of unique classes
    "total_elements": 47,               # Total detections
    "detection_density": 5.41,          # Elements per megapixel
    "average_element_distance": 234.56, # Mean nearest neighbor distance
    "complexity_score": 58.23,          # Computed score
    "complexity_level": "moderate",     # simple/moderate/complex
    "layout_characteristics": {
        "is_dense": True,       # density > 5 elements/megapixel
        "is_diverse": True,     # unique_classes ≥ 10
        "is_structured": False  # avg_distance < 200 pixels
    }
}
```

#### `calculate_quality_metrics(detections, img_width, img_height)`

Detection quality assessment.

##### Overlap Analysis

Measures how many detections overlap (potential errors).

```python
{
    "total_overlapping_pairs": 5,    # Number of pairs with IoU > 0
    "overlap_percentage": 10.64,      # % of detections involved in overlaps
    "average_iou": 0.1234             # Mean IoU of overlapping pairs
}
```

**Overlap Interpretation:**
- **< 5%**: Excellent (minimal overlaps)
- **5% - 15%**: Good (acceptable overlaps)
- **15% - 30%**: Fair (moderate overlaps)
- **> 30%**: Poor (excessive overlaps)

##### Size Consistency

Measures variability in detection sizes.

```python
{
    "coefficient_of_variation": 0.876,  # std/mean of areas
    "consistency_level": "medium"        # high/medium/low
}
```

**Consistency Levels:**
- **high** (CV < 0.5): Very consistent sizes
- **medium** (0.5 ≤ CV < 1.0): Moderate variation
- **low** (CV ≥ 1.0): High variation

##### Detection Quality Score

Overall quality assessment combining overlap and size consistency.

```
score = (1 - min(overlap_% / 100, 1)) × 50 + 
        (1 - min(size_cv, 1)) × 50
```

**Score Interpretation:**
- **80-100**: Excellent quality
- **60-80**: Good quality
- **40-60**: Fair quality
- **< 40**: Poor quality

---

## 📈 Visualization Engine

### services/visualization.py

Generates 8 distinct chart types providing comprehensive visual analysis.

### Chart Types

#### 1. Class Distribution Bar Chart

**Purpose**: Show count of detections per class

**Features:**
- Vertical bars sorted by count (descending)
- Value labels on top of each bar
- Rotated x-axis labels (45°) for readability
- Grid lines for easy counting
- Color: Steel blue

**Code:**
```python
fig, ax = plt.subplots(figsize=(12, 6))
class_counts = sorted(class_metrics.items(), key=lambda x: x[1]['count'], reverse=True)
classes = [item[0] for item in class_counts]
counts = [item[1]['count'] for item in class_counts]

ax.bar(classes, counts, color='steelblue')
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Detection Count by Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

#### 2. Confidence Distribution Histogram

**Purpose**: Show distribution of confidence scores

**Features:**
- 20 bins spanning 0.0 to 1.0
- Red dashed line for mean
- Orange dashed line for median
- Legend with exact values
- Grid for readability

**Code:**
```python
confidences = [d['confidence'] for d in detections]
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
ax.axvline(np.median(confidences), color='orange', linestyle='--', label=f'Median: {np.median(confidences):.3f}')
ax.set_xlabel('Confidence Score')
ax.set_ylabel('Frequency')
ax.set_title('Confidence Distribution')
ax.legend()
plt.tight_layout()
```

#### 3. Spatial Distribution Heatmap

**Purpose**: Visualize where detections are concentrated

**Features:**
- 2D histogram (50x50 bins)
- YlOrRd colormap (yellow → orange → red)
- Colorbar showing density
- Axes in pixel coordinates

**Code:**
```python
x_coords = [d['bbox']['center_x'] for d in detections]
y_coords = [d['bbox']['center_y'] for d in detections]

fig, ax = plt.subplots(figsize=(10, 8))
h = ax.hist2d(x_coords, y_coords, bins=50, cmap='YlOrRd')
plt.colorbar(h[3], ax=ax, label='Density')
ax.set_xlabel('X Coordinate (pixels)')
ax.set_ylabel('Y Coordinate (pixels)')
ax.set_title('Spatial Distribution Heatmap')
plt.tight_layout()
```

#### 4. Confidence by Class Box Plot

**Purpose**: Compare confidence distributions across classes

**Features:**
- Box plot for top 10 classes (by count)
- Shows median, Q1, Q3, outliers
- Sample sizes in x-axis labels
- Light blue boxes with black edges

**Code:**
```python
top_classes = sorted(class_metrics.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
data_to_plot = []
labels = []

for class_name, metrics in top_classes:
    class_detections = [d for d in detections if d['class_name'] == class_name]
    confidences = [d['confidence'] for d in class_detections]
    data_to_plot.append(confidences)
    labels.append(f"{class_name}\n(n={len(confidences)})")

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('Confidence Score')
ax.set_title('Confidence Distribution by Class (Top 10)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

#### 5. Area vs Confidence Scatter Plot

**Purpose**: Examine relationship between size and confidence

**Features:**
- Each point = one detection
- Color-coded by confidence (viridis colormap)
- Colorbar showing confidence scale
- Logarithmic x-axis for better spread

**Code:**
```python
areas = [d['area'] for d in detections]
confidences = [d['confidence'] for d in detections]

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(areas, confidences, c=confidences, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Confidence')
ax.set_xlabel('Detection Area (pixels²)')
ax.set_ylabel('Confidence Score')
ax.set_title('Area vs Confidence')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

#### 6. Quadrant Distribution Pie Chart

**Purpose**: Show spatial distribution by quadrant

**Features:**
- 4 segments (Q1, Q2, Q3, Q4)
- Percentage labels with counts
- Distinct colors per quadrant
- Automatic percentage calculation

**Code:**
```python
quadrants = spatial_metrics['quadrant_distribution']
labels = [f'Q{i}\n({quadrants[f"Q{i}"]} elements)' for i in range(1, 5)]
sizes = [quadrants[f"Q{i}"] for i in range(1, 5)]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Spatial Distribution by Quadrant')
plt.tight_layout()
```

#### 7. Size Distribution Bar Chart

**Purpose**: Show distribution of detection sizes

**Features:**
- 4 categories (tiny, small, medium, large)
- Distinct color per category
- Value labels on bars
- Horizontal grid lines

**Code:**
```python
size_dist = spatial_metrics['size_distribution']
categories = ['tiny', 'small', 'medium', 'large']
counts = [size_dist[cat] for cat in categories]
colors_map = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, counts, color=colors_map)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')
ax.set_xlabel('Size Category')
ax.set_ylabel('Count')
ax.set_title('Size Distribution')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
```

#### 8. Top Classes by Average Confidence

**Purpose**: Identify most confidently detected classes

**Features:**
- Horizontal bars (easier to read class names)
- Top 15 classes only
- Sorted by average confidence
- Value labels at bar ends
- Coral color scheme

**Code:**
```python
class_conf = [(name, metrics['confidence_stats']['mean']) 
              for name, metrics in class_metrics.items()]
class_conf_sorted = sorted(class_conf, key=lambda x: x[1], reverse=True)[:15]
classes = [item[0] for item in class_conf_sorted]
confidences = [item[1] for item in class_conf_sorted]

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(classes, confidences, color='coral')
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{confidences[i]:.3f}', ha='left', va='center', fontsize=9)
ax.set_xlabel('Average Confidence')
ax.set_title('Top 15 Classes by Average Confidence')
ax.invert_yaxis()
plt.tight_layout()
```

### Technical Implementation

```python
def generate_comprehensive_visualizations(
    detections: List[dict],
    class_metrics: dict,
    confidence_metrics: dict,
    spatial_metrics: dict,
    img_width: int,
    img_height: int
) -> dict:
    """
    Generate all visualization types.
    
    Args:
        detections: List of detection dictionaries
        class_metrics: Per-class statistics
        confidence_metrics: Confidence distribution data
        spatial_metrics: Spatial analysis results
        img_width: Original image width
        img_height: Original image height
    
    Returns:
        Dictionary mapping visualization names to base64-encoded PNG images
    """
    visualizations = {}
    
    # Set matplotlib to non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Each visualization wrapped in try-except for isolation
    viz_functions = {
        'class_distribution': lambda: generate_class_distribution(class_metrics),
        'confidence_distribution': lambda: generate_confidence_histogram(detections),
        'spatial_heatmap': lambda: generate_spatial_heatmap(detections),
        'confidence_by_class': lambda: generate_confidence_boxplot(detections, class_metrics),
        'area_vs_confidence': lambda: generate_area_scatter(detections),
        'quadrant_distribution': lambda: generate_quadrant_pie(spatial_metrics),
        'size_distribution': lambda: generate_size_bars(spatial_metrics),
        'top_classes_confidence': lambda: generate_top_confidence(class_metrics)
    }
    
    for viz_name, viz_func in viz_functions.items():
        try:
            fig = viz_func()
            visualizations[viz_name] = fig_to_base64(fig)
            plt.close(fig)  # CRITICAL: Prevents memory leaks
        except Exception as e:
            print(f"Error generating {viz_name}: {e}")
            visualizations[viz_name] = None
    
    return visualizations
```

### Base64 Encoding

```python
from io import BytesIO
import base64

def fig_to_base64(fig) -> str:
    """
    Convert matplotlib figure to base64 data URI.
    
    Args:
        fig: Matplotlib figure object
    
    Returns:
        Base64-encoded string with data URI prefix
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return f"data:image/png;base64,{image_base64}"
```

### Usage in HTML/Frontend

```html
<!-- Display visualizations in web page -->
<div class="visualization-grid">
    <div class="viz-item">
        <h3>Class Distribution</h3>
        <img src="{{ visualizations.class_distribution }}" 
             alt="Class Distribution Chart">
    </div>
    
    <div class="viz-item">
        <h3>Confidence Distribution</h3>
        <img src="{{ visualizations.confidence_distribution }}" 
             alt="Confidence Histogram">
    </div>
    
    <div class="viz-item">
        <h3>Spatial Heatmap</h3>
        <img src="{{ visualizations.spatial_heatmap }}" 
             alt="Spatial Distribution">
    </div>
    
    <!-- ... more visualizations ... -->
</div>
```

---

## 🔧 Services Layer

### services/detection.py

Core detection logic and result processing.

#### `process_detections(result, score_thr=0.3)`

Converts raw MMDetection output to structured format.

**Input**: Raw model result (list of numpy arrays per class)

**Processing Steps:**
1. Iterate through each class's detection array
2. Filter detections by confidence threshold
3. Extract bounding box coordinates
4. Calculate derived metrics (area, aspect ratio, center point)
5. Format as structured dictionaries
6. Sort by confidence (descending)

**Output:**
```python
[
    {
        "class_id": 0,
        "class_name": "paragraph",
        "bbox": {
            "x1": 100.5,
            "y1": 200.3,
            "x2": 500.8,
            "y2": 350.2,
            "width": 400.3,
            "height": 149.9,
            "center_x": 300.65,
            "center_y": 275.25
        },
        "confidence": 0.9234,
        "area": 60005.0,
        "aspect_ratio": 2.67
    },
    // ... more detections    └──────────┘    └────┬─────┘
                                                      │
      ┌───────────────────────────────────────────────┘
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Raw     │───▶│ Process  │───▶│ Calculate│───▶│ Generate │
│  Results │    │Detections│    │ Metrics  │    │  Viz     │
└──────────┘    └──────────┘