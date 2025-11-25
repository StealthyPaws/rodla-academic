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
                    "perturbations": json.dumps(perturbations)
                }
            )
        
        # Should return error or limit to max
        assert response.status_code in [400, 200]
```

#### Integration Tests

```python
# tests/test_integration/test_perturbation_pipeline.py

import pytest
from fastapi.testclient import TestClient
import json
import numpy as np
import cv2
from backend import app

client = TestClient(app)

class TestPerturbationPipeline:
    @pytest.fixture
    def test_image(self, tmp_path):
        """Create realistic test document image."""
        # Create white background
        img = np.ones((500, 700, 3), dtype=np.uint8) * 255
        
        # Add some text-like rectangles
        cv2.rectangle(img, (50, 50), (650, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (50, 150), (650, 200), (0, 0, 0), -1)
        cv2.rectangle(img, (50, 250), (650, 450), (128, 128, 128), 2)
        
        img_path = tmp_path / "test_document.jpg"
        cv2.imwrite(str(img_path), img)
        
        return img_path
    
    def test_full_pipeline_perturb_then_detect(self, test_image):
        """Test complete pipeline: perturb then detect."""
        perturbations = [
            {"type": "defocus", "degree": 1},
            {"type": "speckle", "degree": 1}
        ]
        
        with open(test_image, "rb") as f:
            response = client.post(
                "/api/detect-with-perturbation",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "perturbations": json.dumps(perturbations),
                    "score_thr": "0.3",
                    "generate_visualizations": "false"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check detection results exist
        assert "core_results" in data
        assert "summary" in data["core_results"]
        
        # Check perturbation info exists
        assert "perturbation_info" in data
        assert len(data["perturbation_info"]["applied"]) == 2
    
    def test_compare_clean_vs_perturbed(self, test_image):
        """Compare detection results: clean vs perturbed."""
        # Get clean results
        with open(test_image, "rb") as f:
            clean_response = client.post(
                "/api/detect",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"score_thr": "0.3"}
            )
        
        clean_data = clean_response.json()
        clean_detections = clean_data["core_results"]["summary"]["total_detections"]
        clean_confidence = clean_data["core_results"]["summary"]["average_confidence"]
        
        # Get perturbed results
        with open(test_image, "rb") as f:
            pert_response = client.post(
                "/api/detect-with-perturbation",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "perturbations": json.dumps([{"type": "defocus", "degree": 3}]),
                    "score_thr": "0.3"
                }
            )
        
        pert_data = pert_response.json()
        pert_detections = pert_data["core_results"]["summary"]["total_detections"]
        pert_confidence = pert_data["core_results"]["summary"]["average_confidence"]
        
        # Severe perturbation should affect results
        # (Either fewer detections or lower confidence)
        assert pert_detections <= clean_detections or pert_confidence < clean_confidence
```

### Mocking Dependencies

```python
# tests/conftest.py

import pytest
from unittest.mock import Mock, patch
import numpy as np

@pytest.fixture
def mock_model():
    """Create a mock detection model."""
    model = Mock()
    model.CLASSES = ['paragraph', 'title', 'figure', 'table']
    
    # Mock inference result
    def mock_inference(img):
        # Return fake detections
        return [
            np.array([[100, 100, 500, 300, 0.95]]),  # paragraph
            np.array([[100, 50, 500, 90, 0.90]]),    # title
            np.array([]),                             # figure (none)
            np.array([[200, 350, 450, 480, 0.85]])   # table
        ]
    
    model.side_effect = mock_inference
    return model

@pytest.fixture
def mock_perturbation_functions():
    """Mock perturbation functions for testing."""
    with patch('perturbations.blur.apply_defocus') as mock_defocus, \
         patch('perturbations.noise.apply_speckle') as mock_speckle:
        
        # Return slightly modified image
        mock_defocus.side_effect = lambda img, **kwargs: img + 10
        mock_speckle.side_effect = lambda img, **kwargs: img + 5
        
        yield {
            'defocus': mock_defocus,
            'speckle': mock_speckle
        }
```

---

## 🚢 Deployment

### Development Server

```bash
# Simple development server
python backend.py

# With uvicorn and auto-reload
uvicorn backend:app --reload --host 0.0.0.0 --port 8000

# With specific log level
uvicorn backend:app --reload --log-level debug
```

### Production with Gunicorn

```bash
# Single worker (recommended for GPU)
gunicorn backend:app \
    -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -

# With systemd service
# /etc/systemd/system/rodla-api.service
[Unit]
Description=RoDLA Document Analysis API
After=network.target

[Service]
User=rodla
WorkingDirectory=/opt/rodla-api/deployment
Environment="PATH=/opt/rodla-api/venv/bin"
ExecStart=/opt/rodla-api/venv/bin/gunicorn backend:app \
    -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
Restart=always

[Install]
WantedBy=multi-user.target
```

**Note:** Use `workers=1` for GPU models to avoid CUDA initialization issues across processes.

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p outputs outputs/perturbations

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rodla-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./weights:/app/weights
      - ./perturbation:/app/perturbation  # NEW: Mount perturbation resources
    environment:
      - RODLA_API_KEY=${RODLA_API_KEY}
      - RODLA_BACKGROUND_FOLDER=/app/perturbation/background_image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

**Build and Run:**
```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rodla-api
  labels:
    app: rodla-api
spec:
  replicas: 1  # Single replica for GPU
  selector:
    matchLabels:
      app: rodla-api
  template:
    metadata:
      labels:
        app: rodla-api
    spec:
      containers:
      - name: rodla-api
        image: your-registry/rodla-api:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: RODLA_API_KEY
          valueFrom:
            secretKeyRef:
              name: rodla-secrets
              key: api-key
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            memory: "8Gi"
            cpu: "2"
        volumeMounts:
        - name: outputs
          mountPath: /app/outputs
        - name: weights
          mountPath: /app/weights
        - name: perturbation-resources
          mountPath: /app/perturbation
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: outputs
        persistentVolumeClaim:
          claimName: rodla-outputs-pvc
      - name: weights
        persistentVolumeClaim:
          claimName: rodla-weights-pvc
      - name: perturbation-resources
        persistentVolumeClaim:
          claimName: rodla-perturbation-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rodla-api-service
spec:
  selector:
    app: rodla-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/rodla-api
upstream rodla_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # File upload size (important for large images)
    client_max_body_size 50M;
    
    # Timeouts (important for slow perturbations)
    proxy_connect_timeout 120s;
    proxy_send_timeout 120s;
    proxy_read_timeout 120s;
    
    location / {
        proxy_pass http://rodla_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;
    limit_req zone=api_limit burst=20 nodelay;
    
    # Access logs
    access_log /var/log/nginx/rodla-api-access.log;
    error_log /var/log/nginx/rodla-api-error.log;
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Failures

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Clear GPU memory
nvidia-smi --gpu-reset

# Or in Python
import torch
torch.cuda.empty_cache()
gc.collect()

# Check available memory
nvidia-smi

# Reduce model precision (if supported)
model = model.half()  # Use FP16
```

**Symptom:** `ModuleNotFoundError: No module named 'mmdet'`

**Solution:**
```bash
pip install -U openmim
mim install mmengine mmcv mmdet
```

**Symptom:** `FileNotFoundError: Config file not found`

**Solution:**
```python
# Verify paths in config/settings.py
from pathlib import Path
from config.settings import MODEL_CONFIG, MODEL_WEIGHTS

print(f"Config exists: {Path(MODEL_CONFIG).exists()}")
print(f"Weights exist: {Path(MODEL_WEIGHTS).exists()}")
```

#### 2. Perturbation Failures (NEW)

**Symptom:** `ModuleNotFoundError: No module named 'ocrodeg'`

**Solution:**
```bash
# Install perturbation dependencies
pip install ocrodeg imgaug pyiqa

# Verify installation
python -c "import ocrodeg, imgaug; print('OK')"
```

**Symptom:** `ValueError: background_folder required for 'background' perturbation`

**Solution:**
```bash
# Setup background folder
mkdir -p perturbation/background_image

# Add background images
cp /path/to/textures/*.jpg perturbation/background_image/

# Create index file
python -c "
from content import create_background_file
create_background_file('perturbation/background_image/')
"

# Or provide folder in API call
curl -X POST "http://localhost:8000/api/perturb" \
  -F "file=@image.jpg" \
  -F 'perturbations=[{"type":"background","degree":2}]' \
  -F "background_folder=/path/to/backgrounds"
```

**Symptom:** `ImportError: cannot import name 'apply_rotation' from 'spatial'`

**Solution:**
```bash
# Ensure all perturbation files are copied
ls -la deployment/perturbations/

# Should contain:
# blur.py, content.py, noise.py, inconsistency.py, spatial.py

# Re-copy from RoDLA root if missing
cp /path/to/RoDLA/perturbation/*.py deployment/perturbations/
```

#### 3. Inference Errors

**Symptom:** `RuntimeError: Input type and weight type should be the same`

**Solution:**
```python
# Ensure model and input on same device
model = model.to('cuda')
# or
model = model.to('cpu')

# Check device
print(f"Model device: {next(model.parameters()).device}")
```

**Symptom:** `ValueError: could not broadcast input array`

**Solution:**
```python
# Check image dimensions
from PIL import Image
img = Image.open(image_path)
print(f"Image size: {img.size}")  # (width, height)
print(f"Image mode: {img.mode}")  # Should be RGB or L

# Ensure proper format
img = img.convert('RGB')
```

#### 4. Visualization Errors

**Symptom:** `RuntimeError: main thread is not in main loop`

**Solution:**
```python
# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

**Symptom:** Memory grows with each request

**Solution:**
```python
# Always close figures
fig, ax = plt.subplots()
# ... plotting code ...
plt.savefig(buffer, format='png')
plt.close(fig)  # CRITICAL
plt.close('all')  # Nuclear option

# Monitor memory
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

#### 5. API Errors

**Symptom:** `422 Unprocessable Entity`

**Cause:** Invalid request format

**Solution:**
```bash
# Correct format with proper content type
curl -X POST "http://localhost:8000/api/detect" \
  -H "accept: application/json" \
  -F "file=@image.jpg;type=image/jpeg" \
  -F "score_thr=0.3"

# For perturbations, ensure valid JSON
curl -X POST "http://localhost:8000/api/perturb" \
  -F "file=@image.jpg" \
  -F 'perturbations=[{"type":"defocus","degree":2}]'  # Note single quotes around JSON
```

**Symptom:** `413 Request Entity Too Large`

**Solution:**
```python
# Increase in Nginx
# client_max_body_size 50M;

# Or check FastAPI limits
from fastapi import FastAPI, File, UploadFile

# No explicit limit in FastAPI, controlled by server
```

### Debugging Tips

#### 1. Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Processing image: {filename}")
logger.debug(f"Detections found: {len(detections)}")
logger.debug(f"Perturbations applied: {perturbations}")
```

#### 2. GPU Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# With gpustat (more readable)
pip install gpustat
gpustat -i 1

# Log to file
nvidia-smi -l 1 > gpu_usage.log &
```

#### 3. Memory Profiling

```python
# Install profiler
pip install memory_profiler

# Decorate functions
from memory_profiler import profile

@profile
def detect_objects(...):
    ...

# Run with profiling
python -m memory_profiler backend.py
```

#### 4. Request Timing

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

@app.post("/api/detect")
@timer
async def detect_objects(...):
    ...
```

#### 5. Perturbation Debugging (NEW)

```python
# Test perturbations individually
from perturbations import apply_perturbation
import cv2

image = cv2.imread("test.jpg")

# Test each perturbation
for pert_type in ["defocus", "speckle", "rotation"]:
    for degree in [1, 2, 3]:
        result, success, msg = apply_perturbation(image, pert_type, degree)
        print(f"{pert_type} degree {degree}: {msg}")
        
        if success:
            cv2.imwrite(f"debug_{pert_type}_{degree}.jpg", result)
```

### Health Checks

```python
# Add comprehensive health check endpoint
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check model
    try:
        health_status["components"]["model"] = {
            "status": "ok" if model is not None else "error",
            "loaded": model is not None
        }
    except Exception as e:
        health_status["components"]["model"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check GPU
    try:
        import torch
        health_status["components"]["gpu"] = {
            "status": "ok" if torch.cuda.is_available() else "warning",
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
        }
    except Exception as e:
        health_status["components"]["gpu"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check perturbation dependencies
    try:
        import ocrodeg
        import imgaug
        health_status["components"]["perturbations"] = {
            "status": "ok",
            "dependencies_loaded": True
        }
    except ImportError as e:
        health_status["components"]["perturbations"] = {
            "status": "warning",
            "dependencies_loaded": False,
            "error": str(e)
        }
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        health_status["components"]["disk"] = {
            "status": "ok" if free / total > 0.1 else "warning",
            "free_gb": f"{free / 1024**3:.2f}",
            "total_gb": f"{total / 1024**3:.2f}",
            "usage_percent": f"{(used / total) * 100:.1f}"
        }
    except Exception as e:
        health_status["components"]["disk"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Overall status
    component_statuses = [c["status"] for c in health_status["components"].values()]
    if "error" in component_statuses:
        health_status["status"] = "unhealthy"
    elif "warning" in component_statuses:
        health_status["status"] = "degraded"
    
    return health_status
```

---

## 🤝 Contributing

### Getting Started

1. **Fork the repository**
   ```bash
   # On GitHub, click "Fork"
   git clone https://github.com/yourusername/rodla-api.git
   cd rodla-api
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   # or
   git checkout -b fix/bug-description
   ```

3. **Make your changes**
   - Follow code style guidelines
   - Add tests for new features
   - Update documentation

4. **Run tests**
   ```bash
   pytest
   black .
   isort .
   flake8 .
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Describe your changes
   - Link any related issues
   - Wait for review

### Code Style

```bash
# Install development dependencies
pip install black isort flake8 mypy pytest

# Format code
black . --line-length 100
isort . --profile black

# Check style
flake8 . --max-line-length 100 --ignore E203,W503

# Type checking
mypy . --ignore-missing-imports
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9
        args: ['--line-length=100']
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--ignore=E203,W503']
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

### Adding New Features

#### Adding a New Perturbation Type

1. **Implement the perturbation function** in appropriate module:
   ```python
   # perturbations/blur.py (example)
   
   def apply_new_blur(image, degree, save_path=None):
       """
       Apply new blur effect.
       
       Args:
           image: Input BGR image
           degree: Intensity (1-3)
           save_path: Optional path to save result
       
       Returns:
           Blurred image
       """
       if degree == 0:
           return image
       
       # Implementation here
       degree_value = 2 * degree - 1
       # ... blur logic ...
       
       if save_path:
           cv2.imwrite(save_path, result)
       
       return result
   ```

2. **Export from `__init__.py`**:
   ```python
   # perturbations/__init__.py
   from .blur import apply_defocus, apply_vibration, apply_new_blur  # Add new,
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
    // ... more detections sorted by confidence
]
```

---

### services/processing.py

Result aggregation and persistence.

#### `aggregate_results(...)`

Assembles all analysis components into the final response.

**Parameters:**
- `detections`: Processed detection list
- `core_metrics`: Basic statistics
- `rodla_metrics`: Robustness estimates
- `spatial_metrics`: Spatial analysis
- `class_metrics`: Per-class stats
- `confidence_metrics`: Confidence distribution
- `robustness_indicators`: Stability measures
- `layout_complexity`: Complexity assessment
- `quality_metrics`: Quality scores
- `visualizations`: Base64 charts
- `interpretation`: Human-readable insights
- `file_info`: Image metadata
- `config`: Detection configuration

**Returns**: Complete JSON response dictionary

#### `save_results(results, filename, output_dir)`

Persists results to disk with optimizations.

**Process:**
1. Remove large base64 visualizations from JSON
2. Convert numpy types to Python native types
3. Save JSON with pretty formatting
4. Optionally save visualizations as separate PNG files
5. Return path to saved JSON

---

### 🆕 services/perturbation.py (NEW)

Business logic for image perturbations.

#### `perturb_image_service(image, perturbations, filename, ...)`

Main service function for applying perturbations.

**Parameters:**
```python
image: np.ndarray              # Input BGR image
perturbations: List[dict]      # Perturbation configurations
filename: str                  # Original filename
save_image: bool = False       # Save perturbed image
return_base64: bool = False    # Return as base64
background_folder: str = None  # Background images path
```

**Returns:**
```python
(response_dict, perturbed_image)
```

**Process:**
1. Store original image shape
2. Apply perturbations sequentially via `apply_multiple_perturbations()`
3. Track success/failure for each perturbation
4. Optionally convert to base64
5. Optionally save to disk with descriptive filename
6. Return results and perturbed image

#### `image_to_base64(image)`

Converts OpenCV image to base64 data URI.

**Process:**
1. Encode image to PNG format
2. Convert bytes to base64 string
3. Add data URI prefix
4. Return complete data URI

#### `save_perturbed_image(image, original_filename, perturbations)`

Saves perturbed image with descriptive filename.

**Filename Format:**
```
{original_name}_pert_{pert1type}{degree}_{pert2type}{degree}_{timestamp}.png
```

**Example:**
```
document_pert_defocus2_speckle1_rotation1_20241115_143022.png
```

---

### services/interpretation.py

Human-readable insight generation.

#### `generate_comprehensive_interpretation(...)`

Creates natural language analysis of results.

**Output Sections:**

| Section | Type | Description |
|---------|------|-------------|
| `overview` | string | High-level summary paragraph |
| `top_elements` | string | Most common element description |
| `rodla_analysis` | string | Robustness assessment |
| `layout_complexity` | string | Complexity analysis |
| `key_findings` | list | Important observations (bullet points) |
| `perturbation_assessment` | string | Perturbation effect analysis |
| `recommendations` | list | Actionable suggestions |
| `confidence_summary` | dict | Confidence level breakdown |

**Example Output:**
```python
{
    "overview": """Document Analysis Summary:
Detected 47 layout elements across 12 different classes.
The model achieved an average confidence of 78.2%, 
indicating medium certainty in predictions. The detected 
elements cover 68.5% of the document area.""",

    "top_elements": """The most common elements are:
- Paragraph (15 occurrences, 31.9%)
- Title (8 occurrences, 17.0%)
- Figure (6 occurrences, 12.8%)""",

    "rodla_analysis": """RoDLA Robustness Analysis:
Estimated mPE: 18.45 (low perturbation effect)
Estimated mRD: 87.32 (excellent robustness)
The model shows excellent robustness with minimal 
predicted degradation under perturbations.""",

    "layout_complexity": """Layout Complexity: Moderate
The document has moderate structural complexity with 
12 different element types and 47 total elements.
Detection density: 5.41 elements per megapixel.""",

    "key_findings": [
        "✓ Excellent detection confidence - model is highly certain",
        "✓ High document coverage - most of page contains elements",
        "ℹ Complex document structure with diverse element types",
        "⚠ Some overlapping detections detected (10.6%)"
    ],

    "perturbation_assessment": """Based on confidence 
variability, the model is expected to maintain good 
performance under mild perturbations.""",

    "recommendations": [
        "No specific recommendations - detection quality is good"
    ],

    "confidence_summary": {
        "very_high_count": 15,
        "high_count": 20,
        "medium_count": 10,
        "low_count": 2,
        "very_low_count": 0
    }
}
```

---

## 🛠️ Utilities Reference

### utils/helpers.py

General-purpose mathematical and utility functions.

#### Mathematical Functions

##### `calculate_skewness(data)`

Measures distribution asymmetry.

**Formula:** `mean(((x - μ) / σ)³)`

**Interpretation:**
- **Negative**: Left-skewed (tail on left)
- **Zero**: Symmetric
- **Positive**: Right-skewed (tail on right)

##### `calculate_entropy(values)`

Measures information content/uncertainty.

**Formula:** `-Σ(p × log₂(p))`

**Range:** 0 (certain) to log₂(n) (maximum uncertainty)

##### `calculate_avg_nn_distance(xs, ys)`

Average distance to nearest neighbor.

**Process:**
1. For each point, find closest other point
2. Calculate Euclidean distance
3. Return mean of all distances

##### `calculate_clustering_score(xs, ys)`

Spatial clustering measure.

**Formula:** `1 - (std / mean)` of nearest neighbor distances

**Range:** 0-1 (higher = more clustered)

##### `calculate_iou(bbox1, bbox2)`

Intersection over Union for bounding boxes.

**Formula:** `intersection_area / union_area`

**Range:** 0 (no overlap) to 1 (complete overlap)

#### Utility Functions

##### `calculate_detection_overlaps(detections)`

Finds all overlapping detection pairs.

**Returns:**
```python
{
    'count': int,        # Number of overlapping pairs
    'percentage': float, # % of detections with overlaps
    'avg_iou': float     # Mean IoU of overlaps
}
```

---

### utils/serialization.py

JSON conversion utilities for numpy types.

#### `convert_to_json_serializable(obj)`

Recursively converts numpy types to Python native types.

**Conversions:**

| NumPy Type | Python Type | Example |
|------------|-------------|---------|
| `np.integer` (int8, int32, etc.) | `int` | 42 |
| `np.floating` (float32, float64, etc.) | `float` | 3.14 |
| `np.ndarray` | `list` | [1, 2, 3] |
| `np.bool_` | `bool` | True |

**Implementation:**
```python
def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types for JSON serialization.
    
    Handles:
    - Dictionaries (recursive on values)
    - Lists (recursive on items)
    - NumPy scalars and arrays
    - Native Python types (pass-through)
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
```

---

## ⚠️ Error Handling

### Exception Hierarchy

```
Exception
├── HTTPException (FastAPI)
│   ├── 400 Bad Request
│   │   ├── Invalid file type
│   │   ├── Invalid perturbation config (NEW)
│   │   └── Missing required parameters (NEW)
│   └── 500 Internal Server Error
│       ├── Model not loaded
│       ├── Inference failed
│       ├── Perturbation failed (NEW)
│       └── Processing error
└── Standard Exceptions
    ├── FileNotFoundError
    ├── ValueError
    ├── RuntimeError
    └── ImportError (NEW - perturbation dependencies)
```

### Error Handling Strategy

#### API Level

```python
@app.post("/api/detect")
async def detect_objects(...):
    tmp_path = None
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Main processing logic
        ...
        
    except HTTPException:
        # Re-raise HTTP exceptions unchanged
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
        
    except Exception as e:
        # Handle unexpected errors
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        # Log full traceback
        import traceback
        traceback.print_exc()
        
        # Return structured error response
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )
    
    finally:
        # Always cleanup temp files
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

#### Perturbation Level (NEW)

```python
def apply_perturbation(image, perturbation_type, degree):
    """Apply single perturbation with error handling."""
    try:
        # Validate inputs
        if perturbation_type not in ALL_PERTURBATIONS:
            raise ValueError(f"Invalid perturbation: {perturbation_type}")
        
        if not 1 <= degree <= 3:
            raise ValueError(f"Degree must be 1-3, got: {degree}")
        
        # Apply perturbation
        result_image = ...
        
        return result_image, True, "Success"
        
    except Exception as e:
        # Log error but don't crash
        print(f"Error in {perturbation_type}: {e}")
        return image, False, str(e)
```

### Visualization Error Isolation

Each visualization wrapped individually to prevent cascade failures:

```python
for viz_name, viz_func in visualization_functions.items():
    try:
        visualizations[viz_name] = viz_func()
    except Exception as e:
        print(f"Error generating {viz_name}: {e}")
        visualizations[viz_name] = None
        # Continue with other visualizations
```

### Resource Cleanup

Guaranteed cleanup using try-finally:

```python
def process_image(file):
    tmp_path = None
    perturbed_tmp_path = None
    
    try:
        # Processing logic
        ...
    finally:
        # Always cleanup temp files
        for path in [tmp_path, perturbed_tmp_path]:
            if path and os.path.exists(path):
                os.unlink(path)
```

---

## ⚡ Performance Optimization

### GPU Memory Management

```python
# At startup - clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

# During inference - monitor usage
def detect_with_memory_tracking(model, image):
    initial_memory = torch.cuda.memory_allocated(0)
    
    result = model(image)
    
    peak_memory = torch.cuda.max_memory_allocated(0)
    print(f"Memory used: {(peak_memory - initial_memory) / 1024**2:.2f} MB")
    
    return result
```

### Memory-Efficient Visualizations

```python
def generate_chart():
    fig, ax = plt.subplots()
    
    # ... generate chart ...
    
    # Convert to base64
    base64_str = fig_to_base64(fig)
    
    # CRITICAL: Close figure to free memory
    plt.close(fig)
    
    return base64_str

# After generating all charts
plt.close('all')  # Nuclear option if needed
```

### Response Size Optimization

```python
def save_results(results, filename):
    # Remove large base64 images from saved JSON
    json_results = {
        k: v for k, v in results.items() 
        if k != "visualizations"
    }
    
    # Save lightweight JSON
    json_path = output_dir / f"results_{filename}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save visualizations as separate files
    for viz_name, viz_data in results['visualizations'].items():
        if viz_data:
            save_visualization_to_file(viz_data, f"{filename}_{viz_name}.png")
```

### Lazy Model Loading

```python
# Model loaded once at startup, reused for all requests
@app.on_event("startup")
async def startup_event():
    global model
    model = init_detector(config, weights, device='cuda')
    print("Model loaded and ready")

# Each request uses the same model instance
@app.post("/api/detect")
async def detect_objects(model=Depends(get_model)):
    # No model loading overhead per request
    result = inference_detector(model, image)
```

### Perturbation Optimization (NEW)

```python
def apply_multiple_perturbations(image, perturbations):
    """Apply perturbations efficiently."""
    current_image = image.copy()  # One copy at start
    
    for pert in perturbations:
        # Apply in-place when possible
        current_image, success, msg = apply_perturbation(
            current_image, 
            pert['type'], 
            pert['degree']
        )
        
        if not success:
            # Log but continue with remaining perturbations
            print(f"Perturbation failed: {msg}")
    
    return current_image, results
```

### Performance Benchmarks

| Operation | Time (GPU) | Time (CPU) | Memory (GPU) |
|-----------|------------|------------|--------------|
| Model loading | 10-15s | 20-30s | 6-8 GB |
| Single inference | 0.3-0.5s | 2-5s | +2-3 GB |
| Metrics calculation | 0.1-0.2s | 0.1-0.2s | Minimal |
| Visualization (all 8) | 1-2s | 1-2s | Minimal |
| **Perturbation (single)** | **0.1-0.5s** | **0.1-0.5s** | **Minimal** |
| **Perturbation (3x)** | **0.3-1.5s** | **0.3-1.5s** | **Minimal** |
| **Total (detect only)** | **1.5-3s** | **4-8s** | **8-11 GB** |
| **Total (pert + detect)** | **2-4.5s** | **4.5-9.5s** | **8-11 GB** |

---

## 🔒 Security Considerations

### Current Security Status

| Aspect | Status | Risk | Recommendation |
|--------|--------|------|----------------|
| Authentication | ❌ None | High | Add API key auth |
| CORS | ⚠️ Permissive | Medium | Restrict origins |
| Rate Limiting | ❌ None | Medium | Add throttling |
| Input Validation | ⚠️ Basic | Low | Add size limits |
| Path Handling | ⚠️ Hardcoded | Low | Use env vars |
| **Perturbation Files** | **⚠️ No validation** | **Medium** | **Validate file access** |
| **Background Folder** | **⚠️ User-provided** | **High** | **Sanitize paths** |

### Recommended Security Enhancements

#### 1. API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
import os

API_KEY = os.environ.get("RODLA_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return api_key

# Apply to all endpoints
@app.post("/api/detect")
async def detect_objects(
    ...,
    api_key: str = Depends(verify_api_key)
):
    ...
```

#### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded"}
    )

@app.post("/api/detect")
@limiter.limit("10/minute")
async def detect_objects(...):
    ...

@app.post("/api/perturb")
@limiter.limit("20/minute")  # Higher limit for perturbation-only
async def perturb_image(...):
    ...
```

#### 3. File Size Limits

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Read file content
    content = await file.read()
    
    # Check size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024**2:.1f}MB"
        )
    
    # Continue processing
    ...
```

#### 4. Path Sanitization (NEW - Critical for Perturbations)

```python
from pathlib import Path
import os

def sanitize_path(user_path: str, base_dir: Path) -> Path:
    """
    Sanitize user-provided paths to prevent directory traversal.
    
    Args:
        user_path: User-provided path string
        base_dir: Allowed base directory
    
    Returns:
        Sanitized absolute path
    
    Raises:
        ValueError: If path escapes base directory
    """
    # Resolve to absolute path
    requested_path = Path(user_path).resolve()
    base_dir = base_dir.resolve()
    
    # Check if path is within base directory
    try:
        requested_path.relative_to(base_dir)
    except ValueError:
        raise ValueError(
            f"Access denied: Path outside allowed directory"
        )
    
    # Check if path exists
    if not requested_path.exists():
        raise ValueError(f"Path does not exist: {requested_path}")
    
    return requested_path

# Usage in perturbation endpoint
@app.post("/api/perturb")
async def perturb_image(
    background_folder: Optional[str] = Form(None)
):
    if background_folder:
        try:
            # Sanitize background folder path
            safe_path = sanitize_path(
                background_folder,
                REPO_ROOT / "perturbation"
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
```

#### 5. Input Validation (NEW - Perturbation Configs)

```python
from pydantic import BaseModel, validator

class PerturbationConfig(BaseModel):
    type: str
    degree: int
    
    @validator('type')
    def validate_type(cls, v):
        if v not in ALL_PERTURBATIONS:
            raise ValueError(
                f"Invalid perturbation type. "
                f"Must be one of: {ALL_PERTURBATIONS}"
            )
        return v
    
    @validator('degree')
    def validate_degree(cls, v):
        if not 1 <= v <= 3:
            raise ValueError("Degree must be 1, 2, or 3")
        return v

# Validate perturbation array
def validate_perturbations(perturbations: List[dict]) -> List[PerturbationConfig]:
    if len(perturbations) > MAX_PERTURBATIONS_PER_REQUEST:
        raise HTTPException(
            400,
            f"Maximum {MAX_PERTURBATIONS_PER_REQUEST} perturbations allowed"
        )
    
    validated = []
    for pert in perturbations:
        try:
            validated.append(PerturbationConfig(**pert))
        except Exception as e:
            raise HTTPException(400, f"Invalid perturbation config: {e}")
    
    return validated
```

#### 6. Restricted CORS

```python
# Development
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]

# Production
CORS_ORIGINS = ["https://yourdomain.com", "https://app.yourdomain.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)
```

---

## 🧪 Testing

### Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest fixtures
├── test_api/
│   ├── test_routes.py             # Endpoint tests
│   ├── test_perturbation_routes.py  # 🆕 Perturbation endpoint tests
│   └── test_schemas.py            # Pydantic model tests
├── test_services/
│   ├── test_detection.py          # Detection logic tests
│   ├── test_processing.py         # Processing tests
│   ├── test_visualization.py      # Chart generation tests
│   └── test_perturbation.py       # 🆕 Perturbation service tests
├── test_perturbations/            # 🆕 NEW
│   ├── test_blur.py               # Blur perturbation tests
│   ├── test_noise.py              # Noise perturbation tests
│   ├── test_content.py            # Content perturbation tests
│   ├── test_inconsistency.py      # Inconsistency tests
│   ├── test_spatial.py            # Spatial transform tests
│   └── test_apply.py              # Orchestration tests
├── test_utils/
│   ├── test_helpers.py            # Helper function tests
│   ├── test_metrics.py            # Metrics calculation tests
│   └── test_serialization.py     # Serialization tests
└── test_integration/
    ├── test_full_pipeline.py      # End-to-end tests
    └── test_perturbation_pipeline.py  # 🆕 Perturbation E2E tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_perturbations/test_blur.py

# Run specific test
pytest tests/test_perturbations/test_blur.py::test_defocus_degree_1

# Run with verbose output
pytest -v

# Run only fast tests (no model loading)
pytest -m "not slow"

# Run only perturbation tests
pytest tests/test_perturbations/

# Run with print output
pytest -s
```

### Example Test Cases

#### Testing Perturbation Functions

```python
# tests/test_perturbations/test_blur.py

import pytest
import numpy as np
import cv2
from perturbations.blur import apply_defocus, apply_vibration

class TestDefocus:
    @pytest.fixture
    def sample_image(self):
        """Create a test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_defocus_degree_0(self, sample_image):
        """Degree 0 should return unchanged image."""
        result = apply_defocus(sample_image, degree=0)
        np.testing.assert_array_equal(result, sample_image)
    
    def test_defocus_degree_1(self, sample_image):
        """Degree 1 should apply mild blur."""
        result = apply_defocus(sample_image, degree=1)
        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)
    
    def test_defocus_degree_3(self, sample_image):
        """Degree 3 should apply stronger blur than degree 1."""
        result_1 = apply_defocus(sample_image, degree=1)
        result_3 = apply_defocus(sample_image, degree=3)
        
        # Degree 3 should be more different from original
        diff_1 = np.sum(np.abs(sample_image.astype(float) - result_1.astype(float)))
        diff_3 = np.sum(np.abs(sample_image.astype(float) - result_3.astype(float)))
        assert diff_3 > diff_1
    
    def test_defocus_saves_file(self, sample_image, tmp_path):
        """Test saving to file."""
        save_path = tmp_path / "test_defocus.jpg"
        result = apply_defocus(sample_image, degree=2, save_path=str(save_path))
        
        assert save_path.exists()
        loaded = cv2.imread(str(save_path))
        np.testing.assert_array_equal(loaded, result)

class TestVibration:
    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_vibration_degree_0(self, sample_image):
        result = apply_vibration(sample_image, degree=0)
        np.testing.assert_array_equal(result, sample_image)
    
    def test_vibration_creates_motion_blur(self, sample_image):
        result = apply_vibration(sample_image, degree=2)
        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)
```

#### Testing API Endpoints

```python
# tests/test_api/test_perturbation_routes.py

import pytest
from fastapi.testclient import TestClient
import json
from backend import app

client = TestClient(app)

class TestPerturbationInfo:
    def test_get_perturbation_info(self):
        """Test perturbation info endpoint."""
        response = client.get("/api/perturbations/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_perturbations" in data
        assert data["total_perturbations"] == 12
        assert "categories" in data
        assert len(data["categories"]) == 5

class TestPerturbEndpoint:
    @pytest.fixture
    def sample_image_file(self, tmp_path):
        """Create a sample image file."""
        import numpy as np
        import cv2
        
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), img)
        
        return img_path
    
    def test_perturb_single_perturbation(self, sample_image_file):
        """Test applying single perturbation."""
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/perturb",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "perturbations": json.dumps([{"type": "defocus", "degree": 2}]),
                    "save_image": "false"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert len(data["perturbations_applied"]) == 1
        assert data["perturbations_applied"][0]["type"] == "defocus"
    
    def test_perturb_multiple_perturbations(self, sample_image_file):
        """Test applying multiple perturbations."""
        perturbations = [
            {"type": "defocus", "degree": 2},
            {"type": "speckle", "degree": 1},
            {"type": "rotation", "degree": 1}
        ]
        
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/perturb",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "perturbations": json.dumps(perturbations),
                    "save_image": "false"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert len(data["perturbations_applied"]) == 3
    
    def test_perturb_invalid_type(self, sample_image_file):
        """Test with invalid perturbation type."""
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/perturb",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "perturbations": json.dumps([{"type": "invalid", "degree": 2}])
                }
            )
        
        # Should handle gracefully
        assert response.status_code in [400, 200]
    
    def test_perturb_too_many(self, sample_image_file):
        """Test exceeding max perturbations."""
        perturbations = [
            {"type": "defocus", "degree": i % 3 + 1}
            for i in range(10)  # More than MAX_PERTURBATIONS_PER_REQUEST
        ]
        
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/perturb",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={
                    "perturbations": json.dumps(perturbations)
                    └──────────┘    └────┬─────┘
                                                      │
      ┌───────────────────────────────────────────────┘
      ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Raw     │───▶│ Process  │───▶│ Calculate│───▶│ Generate │
│  Results │    │Detections│    │ Metrics  │    │  Viz     │
└──────────┘    └──────────┘