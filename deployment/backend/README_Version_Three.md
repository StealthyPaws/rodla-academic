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




# Complete Multi-Image Batch Processing Implementation Plan for RoDLA API

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [System Design Decisions](#system-design-decisions)
4. [Component Specifications](#component-specifications)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [API Endpoint Specifications](#api-endpoint-specifications)
7. [Database/Storage Schema](#databasestorage-schema)
8. [Error Handling Strategy](#error-handling-strategy)
9. [Implementation Checklist](#implementation-checklist)
10. [Testing Strategy](#testing-strategy)
11. [Migration Path](#migration-path)

---

## 1. Executive Summary

### 1.1 Project Goal
Add multi-image batch processing capabilities to the existing RoDLA Document Layout Analysis API while maintaining backward compatibility with single-image processing.

### 1.2 Key Requirements Met
- ✅ **Multiple Image Upload**: Users can select and upload 1-300 images in a single request
- ✅ **Async Processing**: Long-running batch jobs processed in background with job ID
- ✅ **Progress Tracking**: Real-time progress via polling endpoint
- ✅ **Flexible Perturbations**: Support both shared and per-image perturbations
- ✅ **Flexible Visualizations**: None, per-image, summary, or both modes
- ✅ **Robust Error Handling**: Continue processing on individual failures
- ✅ **Backward Compatible**: Existing `/api/detect` unchanged
- ✅ **Modular Design**: New endpoints separate from existing code

### 1.3 Technical Stack
- **Framework**: FastAPI with BackgroundTasks
- **Storage**: In-memory dict + JSON file persistence
- **Processing**: Sequential (one image at a time)
- **Upload**: Multipart form-data with List[UploadFile]

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│              (Frontend / API Client / Postman)                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Step 1: POST /api/detect-batch
                 │         (Upload files + config)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API ENDPOINT LAYER                         │
│                   api/routes.py (UPDATED)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  POST /api/detect-batch                                  │  │
│  │  - Validate files (1-300 images)                         │  │
│  │  - Generate unique job_id (UUID4)                        │  │
│  │  - Create job metadata                                   │  │
│  │  - Launch BackgroundTask                                 │  │
│  │  - Return job_id immediately (202 Accepted)             │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GET /api/batch-job/{job_id}                             │  │
│  │  - Return current job status and progress                │  │
│  │  - 200 if exists, 404 if not found                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Step 2: Background processing starts
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKGROUND PROCESSING                         │
│              FastAPI BackgroundTasks + Job Manager              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  BatchJobManager (NEW)                                   │  │
│  │  - Manages in-memory job registry                        │  │
│  │  - Persists jobs to JSON files                           │  │
│  │  - Thread-safe operations                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  process_batch_job() (NEW)                               │  │
│  │  - Loop through each image sequentially                  │  │
│  │  - Apply perturbations (if specified)                    │  │
│  │  - Run detection                                          │  │
│  │  - Calculate metrics                                      │  │
│  │  - Generate visualizations (if requested)                │  │
│  │  - Update job progress after each image                  │  │
│  │  - Handle individual image failures gracefully           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ Step 3: Results stored
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                              │
│                                                                 │
│  outputs/                                                       │
│  ├── jobs/                         ← NEW: Job metadata         │
│  │   ├── job_abc123.json          Job status, progress        │
│  │   └── job_def456.json          Results summary             │
│  │                                                             │
│  ├── batch_20241123_143022/        ← NEW: Batch results       │
│  │   ├── summary.json              Overall batch statistics   │
│  │   ├── image1/                                              │
│  │   │   ├── results.json          Individual results         │
│  │   │   └── visualizations/       8 PNG charts (optional)    │
│  │   ├── image2/                                              │
│  │   │   ├── results.json                                     │
│  │   │   └── visualizations/                                  │
│  │   └── summary_visualizations/   ← Aggregate charts         │
│  │       ├── combined_class_dist.png                          │
│  │       └── ...                                              │
│  │                                                             │
│  └── perturbations/                ← Existing (unchanged)      │
└─────────────────────────────────────────────────────────────────┘
                 │
                 │ Step 4: Client polls for status
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT POLLING                          │
│                                                                 │
│  while job.status != "completed":                              │
│      response = GET /api/batch-job/{job_id}                    │
│      print(f"Progress: {response.processed}/{response.total}") │
│      sleep(2 seconds)                                          │
│                                                                 │
│  # Job complete - fetch results                                │
│  final_response = GET /api/batch-job/{job_id}                  │
│  results = final_response.results                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Hierarchy

```
deployment/
├── backend.py                          ← Main app (minimal changes)
├── config/
│   └── settings.py                     ← UPDATED: Add batch config
├── api/
│   ├── routes.py                       ← UPDATED: Add 2 new endpoints
│   └── schemas.py                      ← UPDATED: Add batch schemas
├── services/
│   ├── detection.py                    ← UNCHANGED
│   ├── processing.py                   ← UPDATED: Add batch functions
│   ├── perturbation.py                 ← UNCHANGED
│   ├── visualization.py                ← UNCHANGED
│   ├── interpretation.py               ← UNCHANGED
│   └── batch_job_manager.py            ← NEW: Job management
└── outputs/
    ├── jobs/                           ← NEW: Job metadata storage
    └── batch_*/                        ← NEW: Batch result directories
```

---

## 3. System Design Decisions

### 3.1 Async Processing with Job IDs

#### Why Async?
- **User Experience**: 300 images × 3 seconds = 15 minutes. Can't block HTTP request.
- **HTTP Timeout**: Most proxies/browsers timeout after 60-120 seconds.
- **Scalability**: Allows concurrent batch submissions (future enhancement).

#### Why FastAPI BackgroundTasks?
- **Simplicity**: Built-in, no external dependencies (Redis, Celery).
- **Sufficient**: For single-worker, single-GPU setup.
- **Easy Migration**: Can upgrade to Celery later without changing API contract.

#### Job ID Generation
```python
import uuid

job_id = str(uuid.uuid4())  # e.g., "3fa85f64-5717-4562-b3fc-2c963f66afa6"
```
- **Unique**: Collision probability negligible
- **URL-safe**: No special characters
- **Unguessable**: Cannot enumerate other users' jobs

### 3.2 Storage Strategy: In-Memory + JSON Files

#### In-Memory Registry
```python
# Global dictionary (thread-safe with locks)
job_registry = {
    "job_abc123": {
        "job_id": "abc123",
        "status": "processing",  # queued, processing, completed, failed
        "progress": {"current": 5, "total": 10},
        "results": [...],
        # ... more fields
    }
}
```

**Advantages:**
- ✅ Fast reads (no disk I/O)
- ✅ Simple implementation
- ✅ Sufficient for single-instance deployment

**Disadvantages:**
- ❌ Lost on restart (mitigated by JSON persistence)
- ❌ Not shared across multiple API instances (future concern)

#### JSON File Persistence
```python
# Saved to: outputs/jobs/job_abc123.json
{
    "job_id": "abc123",
    "status": "completed",
    "created_at": "2024-11-23T14:30:22",
    "updated_at": "2024-11-23T14:45:56",
    "config": {...},
    "progress": {...},
    "results": [...]
}
```

**When to Persist:**
- ✅ Job creation (initial state)
- ✅ After each image processed (progress update)
- ✅ Job completion (final state)
- ✅ On error (for debugging)

**Recovery on Restart:**
- Load all JSON files from `outputs/jobs/` into memory
- Mark incomplete jobs as "failed" with note "Server restarted"

### 3.3 Sequential Processing

#### Processing Loop
```python
for i, image_file in enumerate(uploaded_files):
    try:
        # 1. Save temp file
        # 2. Apply perturbations (if any)
        # 3. Run detection
        # 4. Calculate metrics
        # 5. Generate visualizations (if requested)
        # 6. Save results
        # 7. Update job progress
        # 8. Persist to JSON
    except Exception as e:
        # Log error, mark image as failed, continue
        pass
```

#### Why Sequential, Not Parallel?
- **GPU Limitation**: Single GPU, MMDetection not designed for concurrent inference
- **Memory Safety**: Avoids OOM errors from loading multiple images
- **Predictability**: Easier to debug, trace, and test
- **Progress Tracking**: Simple to implement (current = i+1)

#### Future Parallelization (Not in MVP)
- Could process on multiple GPUs if available
- Would require semaphore to limit concurrent workers
- Would need more complex progress tracking

### 3.4 Perturbation Handling

#### Format Support
```python
# Option 1: Single set for all images
perturbations = [{"type": "defocus", "degree": 2}]
# Applied to image1, image2, ..., imageN

# Option 2: Per-image perturbations
perturbations = [
    [{"type": "defocus", "degree": 2}],  # For image1
    [{"type": "speckle", "degree": 1}],  # For image2
    # ... must match number of images
]
```

#### Detection Logic
```python
if perturbations:
    if isinstance(perturbations[0], list):
        # Per-image mode
        if len(perturbations) != len(files):
            raise ValueError("Perturbations count must match files count")
        pert_for_image = perturbations[i]
    else:
        # Shared mode
        pert_for_image = perturbations
```

#### Validation
- **Type Check**: Must be list (not dict, not string)
- **Length Check**: If nested, must match file count
- **Perturbation Validation**: Reuse existing validation from `/api/perturb`

### 3.5 Visualization Modes

#### Four Modes
```python
visualization_mode: str = Form("none")
# Options: "none", "per_image", "summary", "both"
```

#### Mode Behavior

| Mode | Per-Image Charts | Summary Charts | Use Case |
|------|------------------|----------------|----------|
| `none` | ❌ No | ❌ No | Fast processing, no analysis |
| `per_image` | ✅ 8 charts × N images | ❌ No | Detailed per-image inspection |
| `summary` | ❌ No | ✅ 8 aggregate charts | Quick batch overview |
| `both` | ✅ Per-image | ✅ Summary | Complete analysis |

#### Storage Impact

**For 100 images with `per_image` mode:**
```
batch_20241123_143022/
├── image001/
│   └── visualizations/  (8 PNGs × ~200KB = 1.6 MB)
├── image002/
│   └── visualizations/  (8 PNGs × ~200KB = 1.6 MB)
...
├── image100/
│   └── visualizations/  (8 PNGs × ~200KB = 1.6 MB)
└── summary_visualizations/  (optional, +1.6 MB)

Total: ~160 MB (800 PNG files)
```

#### Summary Statistics Aggregation
```python
# Aggregate across all images
summary_stats = {
    "total_images": 100,
    "total_detections": 4752,
    "average_detections_per_image": 47.52,
    "confidence_distribution": {
        "mean": 0.78,
        "std": 0.12,
        "min": 0.30,
        "max": 0.99
    },
    "class_distribution": {
        "paragraph": 1523,
        "title": 845,
        # ... aggregated across all images
    },
    # ... more aggregate metrics
}
```

#### Summary Visualizations
```python
# Example: Combined class distribution chart
# X-axis: Classes
# Y-axis: Total count across all images
# Bar colors: Same as single-image charts

# 8 Summary Charts:
1. combined_class_distribution.png
2. combined_confidence_histogram.png
3. average_confidence_by_class.png
4. detection_count_per_image.png  (NEW)
5. confidence_trend_across_images.png  (NEW)
6. processing_time_per_image.png  (NEW)
7. success_failure_summary.png  (NEW)
8. top_classes_across_batch.png
```

### 3.6 Error Handling Philosophy

#### Per-Image Error Isolation
```python
for image in images:
    try:
        result = process_single_image(image)
        results.append({"status": "success", "data": result})
    except Exception as e:
        results.append({
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        # Continue to next image
```

#### Job-Level Status
```python
if all_images_successful:
    job_status = "completed"
elif any_images_successful:
    job_status = "partial"  # Some succeeded, some failed
else:
    job_status = "failed"  # All images failed
```

#### Error Categories

| Error Type | HTTP Status | Job Status | Action |
|------------|-------------|------------|--------|
| Invalid file format | 400 | Not created | Return error immediately |
| Too many files (>300) | 400 | Not created | Return error immediately |
| Model inference error | 500 | Partial | Mark image failed, continue |
| Perturbation error | 500 | Partial | Mark image failed, continue |
| Disk space exhausted | 500 | Failed | Stop job, return error |
| Out of memory | 500 | Partial | Mark image failed, GC, continue |

---

## 4. Component Specifications

### 4.1 BatchJobManager Class

**Purpose:** Central manager for all batch job operations.

**Location:** `services/batch_job_manager.py` (NEW FILE)

#### Class Structure
```python
import threading
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import uuid

class BatchJobManager:
    """
    Thread-safe manager for batch processing jobs.
    
    Responsibilities:
    - Create new jobs
    - Update job progress
    - Retrieve job status
    - Persist jobs to disk
    - Load jobs from disk (on startup)
    """
    
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory registry
        self._jobs: Dict[str, dict] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load existing jobs on init
        self._load_jobs_from_disk()
    
    def create_job(self, config: dict) -> str:
        """Create new job and return job_id"""
        
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID"""
        
    def update_job_progress(self, job_id: str, current: int, total: int):
        """Update processing progress"""
        
    def add_job_result(self, job_id: str, image_name: str, result: dict):
        """Add result for single image"""
        
    def mark_job_completed(self, job_id: str):
        """Mark job as completed"""
        
    def mark_job_failed(self, job_id: str, error: str):
        """Mark job as failed"""
        
    def _persist_job(self, job_id: str):
        """Save job to JSON file"""
        
    def _load_jobs_from_disk(self):
        """Load all jobs from disk on startup"""
```

#### Job Schema (In-Memory & Persisted)
```python
{
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "status": "processing",  # queued, processing, completed, partial, failed
    "created_at": "2024-11-23T14:30:22.123456",
    "updated_at": "2024-11-23T14:35:45.789012",
    "started_at": "2024-11-23T14:30:23.456789",
    "completed_at": None,  # Set when finished
    
    # Input configuration
    "config": {
        "total_images": 10,
        "score_threshold": 0.3,
        "visualization_mode": "summary",
        "perturbations": [...],  # or None
        "perturbation_mode": "shared",  # or "per_image"
        "save_json": True,
        "filenames": ["img1.jpg", "img2.jpg", ...]
    },
    
    # Progress tracking
    "progress": {
        "current": 5,      # Images processed so far
        "total": 10,       # Total images
        "percentage": 50.0,
        "successful": 4,   # Successfully processed
        "failed": 1,       # Failed to process
        "processing_times": [2.3, 1.8, 3.1, 2.0, 2.5]  # Seconds per image
    },
    
    # Results (per image)
    "results": [
        {
            "image": "img1.jpg",
            "status": "success",
            "processing_time": 2.3,
            "detections_count": 47,
            "average_confidence": 0.82,
            "result_path": "outputs/batch_xxx/img1/results.json"
        },
        {
            "image": "img2.jpg",
            "status": "failed",
            "error": "Model inference failed: CUDA out of memory",
            "traceback": "..."
        },
        # ... more results
    ],
    
    # Batch output directory
    "output_dir": "outputs/batch_20241123_143022",
    
    # Summary statistics (filled after completion)
    "summary": {
        "total_detections": 456,
        "average_confidence": 0.79,
        "processing_time_total": 23.5,
        "processing_time_average": 2.35,
        # ... more aggregated stats
    }
}
```

#### Thread Safety Implementation
```python
def update_job_progress(self, job_id: str, current: int, total: int):
    """Thread-safe progress update"""
    with self._lock:
        if job_id not in self._jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self._jobs[job_id]
        job["progress"]["current"] = current
        job["progress"]["total"] = total
        job["progress"]["percentage"] = (current / total) * 100
        job["updated_at"] = datetime.now().isoformat()
        
        # Persist to disk
        self._persist_job(job_id)
```

### 4.2 Batch Processing Function

**Purpose:** Background task that processes all images in a job.

**Location:** `services/processing.py` (ADD NEW FUNCTION)

#### Function Signature
```python
def process_batch_job(
    job_id: str,
    uploaded_files: List[UploadFile],
    config: dict,
    job_manager: BatchJobManager
):
    """
    Process a batch of images in the background.
    
    Args:
        job_id: Unique job identifier
        uploaded_files: List of UploadFile objects
        config: Job configuration dict
        job_manager: BatchJobManager instance
    
    Returns:
        None (updates job status via job_manager)
    """
```

#### Processing Logic Flow
```python
async def process_batch_job(job_id, uploaded_files, config, job_manager):
    # 1. Mark job as started
    job_manager.mark_job_started(job_id)
    
    # 2. Create batch output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_DIR / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Process each image sequentially
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # 3.1 Save uploaded file to temp location
            temp_path = save_temp_file(uploaded_file)
            
            # 3.2 Read image
            image = cv2.imread(temp_path)
            
            # 3.3 Apply perturbations (if configured)
            if config.get("perturbations"):
                perturbations = get_perturbations_for_image(
                    config, i
                )
                image, pert_result = apply_perturbations(
                    image, perturbations, ...
                )
            
            # 3.4 Save perturbed image to temp location
            perturbed_temp_path = save_perturbed_temp(image)
            
            # 3.5 Run detection
            result, img_w, img_h = run_inference(perturbed_temp_path)
            detections = process_detections(result, config["score_threshold"])
            
            # 3.6 Calculate metrics
            results_dict = create_comprehensive_results(
                detections,
                img_w,
                img_h,
                uploaded_file.filename,
                config["score_threshold"],
                generate_viz=(config["visualization_mode"] in ["per_image", "both"])
            )
            
            # 3.7 Save individual result
            image_dir = batch_dir / f"image_{i+1:03d}"
            image_dir.mkdir(exist_ok=True)
            save_individual_result(results_dict, image_dir)
            
            # 3.8 Add to job results
            job_manager.add_job_result(job_id, uploaded_file.filename, {
                "status": "success",
                "processing_time": processing_time,
                "detections_count": len(detections),
                "average_confidence": results_dict["core_results"]["summary"]["average_confidence"],
                "result_path": str(image_dir / "results.json")
            })
            
            # 3.9 Update progress
            job_manager.update_job_progress(
                job_id, 
                current=i+1, 
                total=len(uploaded_files)
            )
            
            # 3.10 Cleanup temp files
            cleanup_temp_files(temp_path, perturbed_temp_path)
            
        except Exception as e:
            # Handle error for this image
            job_manager.add_job_result(job_id, uploaded_file.filename, {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            # Update progress (still count as processed)
            job_manager.update_job_progress(
                job_id,
                current=i+1,
                total=len(uploaded_files)
            )
    
    # 4. Generate summary statistics
    if config["visualization_mode"] in ["summary", "both"]:
        summary_stats, summary_viz = generate_batch_summary(
            job_id, job_manager
        )
        save_summary(batch_dir, summary_stats, summary_viz)
    
    # 5. Mark job as completed
    job_manager.mark_job_completed(job_id)
```

#### Helper Functions
```python
def get_perturbations_for_image(config: dict, image_index: int) -> List[dict]:
    """
    Get perturbations for specific image based on mode.
    
    Args:
        config: Job config with perturbations
        image_index: Current image index (0-based)
    
    Returns:
        List of perturbation dicts for this image
    """
    perturbations = config.get("perturbations")
    if not perturbations:
        return []
    
    if config["perturbation_mode"] == "per_image":
        return perturbations[image_index]
    else:  # shared
        return perturbations


def save_individual_result(results_dict: dict, image_dir: Path):
    """
    Save individual image results to directory.
    
    Structure:
    image_dir/
    ├── results.json           (Full results)
    └── visualizations/        (If generated)
        ├── class_distribution.png
        ├── confidence_histogram.png
        └── ...
    """
    # Save JSON (without base64 visualizations)
    json_path = image_dir / "results.json"
    save_results(results_dict, json_path.stem, save_visualizations=True)
    
    # Save visualizations if present
    if results_dict.get("visualizations"):
        viz_dir = image_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for viz_name, viz_base64 in results_dict["visualizations"].items():
            if viz_base64:
                viz_path = viz_dir / f"{viz_name}.png"
                save_base64_image(viz_base64, viz_path)


def generate_batch_summary(job_id: str, job_manager: BatchJobManager) -> tuple:
    """
    Generate aggregate statistics and visualizations for entire batch.
    
    Returns:
        (summary_stats_dict, summary_visualizations_dict)
    """
    job = job_manager.get_job(job_id)
    
    # Aggregate metrics across all successful results
    all_detections = []
    all_confidences = []
    class_counts = {}
    
    for result in job["results"]:
        if result["status"] == "success":
            # Load individual result file
            result_data = load_json(result["result_path"])
            
            # Collect detections
            all_detections.extend(result_data["all_detections"])
            
            # Collect confidences
            for det in result_data["all_detections"]:
                all_confidences.append(det["confidence"])
            
            # Aggregate class counts
            for class_name, class_data in result_data["class_analysis"].items():
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += class_data["count"]
    
    # Calculate summary statistics
    summary_stats = {
        "total_images": job["config"]["total_images"],
        "successful_images": job["progress"]["successful"],
        "failed_images": job["progress"]["failed"],
        "total_detections": len(all_detections),
        "average_detections_per_image": len(all_detections) / max(job["progress"]["successful"], 1),
        "average_confidence": np.mean(all_confidences) if all_confidences else 0,
        "confidence_std": np.std(all_confidences) if all_confidences else 0,
        "class_distribution": class_counts,
        "processing_time_total": sum(job["progress"]["processing_times"]),
        "processing_time_average": np.mean(job["progress"]["processing_times"]) if job["progress"]["processing_times"] else 0,
        # ... more aggregated metrics
    }
    
    # Generate summary visualizations
    summary_viz = generate_summary_visualizations(
        summary_stats,
        all_detections,
        job["progress"]["processing_times"]
    )
    
    return summary_stats, summary_viz


def generate_summary_visualizations(
    summary_stats: dict,
    all_detections: List[dict],
    processing_times: List[float]
) -> dict:
    """
    Generate 8 summary charts for the entire batch.
    
    Returns:
        Dict mapping chart names to base64 PNGs
    """
    visualizations = {}
    
    # 1. Combined class distribution (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    classes = list(summary_stats["class_distribution"].keys())
    counts = list(summary_stats["class_distribution"].values())
    ax.bar(classes, counts, color='steelblue')
    ax.set_title("Class Distribution Across All Images")
    ax.set_xlabel("Class")
    ax.set_ylabel("Total Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    visualizations['combined_class_distribution'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 2. Combined confidence histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    all_confidences = [d['confidence'] for d in all_detections]
    ax.hist(all_confidences, bins=20, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(all_confidences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_confidences):.3f}')
    ax.set_title("Confidence Distribution Across All Images")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    visualizations['combined_confidence_histogram'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 3. Detection count per image (line chart)
    # Shows trend of detections across images
    
    # 4. Confidence trend across images
    # Shows average confidence per image
    
    # 5. Processing time per image (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    image_indices = list(range(1, len(processing_times) + 1))
    ax.bar(image_indices, processing_times, color='coral')
    ax.axhline(np.mean(processing_times), color='red', linestyle='--',
               label=f'Average: {np.mean(processing_times):.2f}s')
    ax.set_title("Processing Time Per Image")
    ax.set_xlabel("Image Number")
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    plt.tight_layout()
    visualizations['processing_time_per_image'] = fig_to_base64(fig)
    plt.close(fig)
    
    # 6. Success/Failure summary (pie chart)
    # 7. Top classes across batch
    # 8. Average confidence by class
    
    return visualizations
```

---

### 4.3 API Route Updates

**Location:** `api/routes.py` (ADD TWO NEW ENDPOINTS)

#### Endpoint 1: POST /api/detect-batch

```python
@router.post("/api/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(..., description="1-300 image files"),
    score_thr: str = Form("0.3", description="Confidence threshold (0-1)"),
    perturbations: Optional[str] = Form(None, description="JSON: single list or list of lists"),
    visualization_mode: str = Form("none", description="none|per_image|summary|both"),
    save_json: str = Form("true", description="Save results to disk"),
    background_folder: Optional[str] = Form(None, description="Background images folder"),
    background_tasks: BackgroundTasks = None
):
    """
    Process multiple images in batch mode with async processing.
    
    **Request Parameters:**
    - **files**: 1-300 image files (multipart/form-data)
    - **score_thr**: Confidence threshold (0.0-1.0), default 0.3
    - **perturbations**: Optional JSON string
        - Single list: `[{"type":"defocus","degree":2}]` (applied to all)
        - List of lists: `[[...], [...]]` (per-image, must match file count)
    - **visualization_mode**: Visualization generation mode
        - `none`: No visualizations (fastest)
        - `per_image`: Generate 8 charts per image (slowest)
        - `summary`: Generate 8 aggregate charts (moderate)
        - `both`: Per-image + summary (comprehensive)
    - **save_json**: Save results to disk (default: true)
    - **background_folder**: Path to background images (for 'background' perturbation)
    
    **Response (202 Accepted):**
    ```json
    {
        "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "status": "queued",
        "message": "Batch processing started",
        "total_images": 10,
        "status_endpoint": "/api/batch-job/3fa85f64-5717-4562-b3fc-2c963f66afa6"
    }
    ```
    
    **Poll the status endpoint to track progress.**
    """
    
    try:
        # 1. Validate file count
        if not files or len(files) == 0:
            raise HTTPException(400, "At least one file is required")
        
        if len(files) > 300:
            raise HTTPException(400, f"Maximum 300 files allowed, got {len(files)}")
        
        # 2. Validate file types
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(400, f"File {file.filename} is not an image")
        
        # 3. Parse and validate score threshold
        try:
            score_threshold = float(score_thr)
            if not 0 <= score_threshold <= 1:
                raise ValueError("Must be between 0 and 1")
        except ValueError as e:
            raise HTTPException(400, f"Invalid score_thr: {e}")
        
        # 4. Parse and validate perturbations
        perturbation_config = None
        perturbation_mode = "shared"  # or "per_image"
        
        if perturbations:
            try:
                pert_data = json.loads(perturbations)
                
                if not isinstance(pert_data, list):
                    raise ValueError("Perturbations must be a list")
                
                # Check if per-image mode (list of lists)
                if pert_data and isinstance(pert_data[0], list):
                    perturbation_mode = "per_image"
                    
                    # Validate count matches
                    if len(pert_data) != len(files):
                        raise ValueError(
                            f"Per-image perturbations count ({len(pert_data)}) "
                            f"must match file count ({len(files)})"
                        )
                    
                    # Validate each sub-list
                    for i, pert_list in enumerate(pert_data):
                        if not isinstance(pert_list, list):
                            raise ValueError(f"Perturbations[{i}] must be a list")
                        # TODO: Validate each perturbation config
                else:
                    # Shared mode - validate single list
                    # TODO: Validate perturbation configs
                    pass
                
                perturbation_config = pert_data
                
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid JSON in perturbations parameter")
            except ValueError as e:
                raise HTTPException(400, str(e))
        
        # 5. Validate visualization mode
        valid_viz_modes = ["none", "per_image", "summary", "both"]
        if visualization_mode not in valid_viz_modes:
            raise HTTPException(
                400, 
                f"Invalid visualization_mode. Must be one of: {valid_viz_modes}"
            )
        
        # 6. Create job configuration
        job_config = {
            "total_images": len(files),
            "score_threshold": score_threshold,
            "visualization_mode": visualization_mode,
            "perturbations": perturbation_config,
            "perturbation_mode": perturbation_mode,
            "save_json": save_json.lower() == "true",
            "background_folder": background_folder,
            "filenames": [f.filename for f in files]
        }
        
        # 7. Create job in job manager
        job_manager = get_job_manager()  # Singleton instance
        job_id = job_manager.create_job(job_config)
        
        # 8. Save uploaded files to temporary location
        # (BackgroundTask needs file paths, not UploadFile objects)
        temp_file_paths = []
        for file in files:
            temp_path = save_uploaded_file_temp(file)
            temp_file_paths.append(temp_path)
        
        # 9. Start background processing
        background_tasks.add_task(
            process_batch_job,
            job_id=job_id,
            temp_file_paths=temp_file_paths,
            config=job_config,
            job_manager=job_manager
        )
        
        # 10. Return job ID immediately
        return JSONResponse(
            content={
                "job_id": job_id,
                "status": "queued",
                "message": f"Batch processing started for {len(files)} images",
                "total_images": len(files),
                "status_endpoint": f"/api/batch-job/{job_id}",
                "estimated_time_seconds": len(files) * 3  # Rough estimate
            },
            status_code=202  # Accepted
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )
```

#### Endpoint 2: GET /api/batch-job/{job_id}

```python
@router.get("/api/batch-job/{job_id}")
async def get_batch_job_status(job_id: str):
    """
    Get status and results of a batch processing job.
    
    **Path Parameters:**
    - **job_id**: Unique job identifier returned by /api/detect-batch
    
    **Response States:**
    
    **1. Queued (just started):**
    ```json
    {
        "job_id": "abc123",
        "status": "queued",
        "created_at": "2024-11-23T14:30:22",
        "progress": {
            "current": 0,
            "total": 10,
            "percentage": 0.0
        }
    }
    ```
    
    **2. Processing (in progress):**
    ```json
    {
        "job_id": "abc123",
        "status": "processing",
        "progress": {
            "current": 5,
            "total": 10,
            "percentage": 50.0,
            "successful": 4,
            "failed": 1
        },
        "results": [
            {"image": "img1.jpg", "status": "success", ...},
            {"image": "img2.jpg", "status": "success", ...},
            {"image": "img3.jpg", "status": "success", ...},
            {"image": "img4.jpg", "status": "success", ...},
            {"image": "img5.jpg", "status": "failed", "error": "..."}
        ],
        "estimated_time_remaining_seconds": 15
    }
    ```
    
    **3. Completed (finished successfully):**
    ```json
    {
        "job_id": "abc123",
        "status": "completed",
        "created_at": "2024-11-23T14:30:22",
        "completed_at": "2024-11-23T14:45:56",
        "progress": {
            "current": 10,
            "total": 10,
            "percentage": 100.0,
            "successful": 10,
            "failed": 0
        },
        "results": [...],  // All 10 image results
        "summary": {
            "total_detections": 456,
            "average_confidence": 0.79,
            "processing_time_total": 23.5,
            ...
        },
        "output_dir": "outputs/batch_20241123_143022"
    }
    ```
    
    **4. Partial (some images failed):**
    ```json
    {
        "job_id": "abc123",
        "status": "partial",
        "progress": {
            "current": 10,
            "total": 10,
            "percentage": 100.0,
            "successful": 8,
            "failed": 2
        },
        "results": [...],
        "summary": {...}
    }
    ```
    
    **5. Failed (all images failed or critical error):**
    ```json
    {
        "job_id": "abc123",
        "status": "failed",
        "error": "Critical error description",
        "progress": {...}
    }
    ```
    """
    
    try:
        # Get job from manager
        job_manager = get_job_manager()
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        
        # Return current job state
        return JSONResponse(content=job)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )
```

### 4.4 Schema Updates

**Location:** `api/schemas.py` (ADD NEW MODELS)

```python
# ============================================================================
# BATCH PROCESSING SCHEMAS
# ============================================================================

class BatchJobConfig(BaseModel):
    """Configuration for a batch processing job."""
    total_images: int = Field(..., ge=1, le=300)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    visualization_mode: str = Field("none", pattern="^(none|per_image|summary|both)$")
    perturbations: Optional[List] = None  # Can be List[dict] or List[List[dict]]
    perturbation_mode: str = Field("shared", pattern="^(shared|per_image)$")
    save_json: bool = True
    background_folder: Optional[str] = None
    filenames: List[str]


class BatchJobProgress(BaseModel):
    """Progress tracking for batch job."""
    current: int = Field(..., ge=0)
    total: int = Field(..., ge=1)
    percentage: float = Field(..., ge=0.0, le=100.0)
    successful: int = Field(0, ge=0)
    failed: int = Field(0, ge=0)
    processing_times: List[float] = []


class BatchImageResult(BaseModel):
    """Result for a single image in batch."""
    image: str
    status: str  # "success" or "failed"
    processing_time: Optional[float] = None
    detections_count: Optional[int] = None
    average_confidence: Optional[float] = None
    result_path: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


class BatchJobSummary(BaseModel):
    """Aggregate statistics for completed batch."""
    total_images: int
    successful_images: int
    failed_images: int
    total_detections: int
    average_detections_per_image: float
    average_confidence: float
    confidence_std: float
    class_distribution: Dict[str, int]
    processing_time_total: float
    processing_time_average: float


class BatchJobStatus(BaseModel):
    """Complete status of a batch job."""
    job_id: str
    status: str  # queued, processing, completed, partial, failed
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: BatchJobConfig
    progress: BatchJobProgress
    results: List[BatchImageResult]
    summary: Optional[BatchJobSummary] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None


class BatchJobCreateResponse(BaseModel):
    """Response when creating a new batch job."""
    job_id: str
    status: str = "queued"
    message: str
    total_images: int
    status_endpoint: str
    estimated_time_seconds: int
```

### 4.5 Configuration Updates

**Location:** `config/settings.py` (ADD NEW SETTINGS)

```python
# ============================================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================================

# Job storage directory
JOBS_DIR = OUTPUT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Batch processing limits
MAX_BATCH_SIZE = 300  # Maximum images per batch
MIN_BATCH_SIZE = 1    # Minimum images per batch

# Job retention
JOB_RETENTION_HOURS = 48  # Keep jobs for 48 hours
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
```

---

## 5. Data Flow Diagrams

### 5.1 Request Flow: Submit Batch Job

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     │ 1. POST /api/detect-batch
     │    files: [img1, img2, img3, ..., img10]
     │    score_thr: "0.3"
     │    visualization_mode: "summary"
     │    perturbations: '[{"type":"defocus","degree":2}]'
     ▼
┌─────────────────────────────────────┐
│   FastAPI Router (routes.py)       │
│   @router.post("/api/detect-batch") │
└────┬────────────────────────────────┘
     │
     │ 2. Validate inputs
     │    - Check file count (1-300)
     │    - Validate file types (images only)
     │    - Parse score_thr (0.0-1.0)
     │    - Parse perturbations JSON
     │    - Validate visualization_mode
     ▼
┌─────────────────────────────────────┐
│   BatchJobManager                   │
│   .create_job(config)               │
└────┬────────────────────────────────┘
     │
     │ 3. Generate job_id (UUID4)
     │    Create job dict
     │    Save to memory: self._jobs[job_id] = {...}
     │    Persist to disk: jobs/job_abc123.json
     ▼
┌─────────────────────────────────────┐
│   Save Uploaded Files to Temp      │
│   /tmp/upload_img1_xyz.jpg          │
│   /tmp/upload_img2_abc.jpg          │
│   ...                                │
└────┬────────────────────────────────┘
     │
     │ 4. Launch BackgroundTask
     │    process_batch_job(job_id, temp_paths, config)
     ▼
┌─────────────────────────────────────┐
│   FastAPI BackgroundTasks           │
│   (Runs in separate thread)         │
└────┬────────────────────────────────┘
     │
     │ 5. Return 202 Accepted immediately
     ▼
┌──────────┐
│  Client  │  Receives:
└──────────┘  {
               "job_id": "abc123",
               "status": "queued",
               "status_endpoint": "/api/batch-job/abc123"
              }
```

### 5.2 Background Processing Flow

```
┌─────────────────────────────────────┐
│   BackgroundTask Started            │
│   process_batch_job()               │
└────┬────────────────────────────────┘
     │
     │ 1. Mark job as "processing"
     │    job_manager.mark_job_started(job_id)
     ▼
┌─────────────────────────────────────┐
│   Create Batch Output Directory     │
│   outputs/batch_20241123_143022/    │
└────┬────────────────────────────────┘
     │
     │ 2. Loop: for i, temp_path in enumerate(temp_file_paths):
     ▼
┌──────────────────────────────────────────────────────────┐
│   Process Image i                                        │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │ a. Load image: cv2.imread(temp_path)            │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ b. Apply perturbations (if configured)          │  │
│   │    - Get perturbations for this image           │  │
│   │    - Call perturb_image_service()               │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ c. Run detection                                 │  │
│   │    - result, w, h = run_inference(image_path)   │  │
│   │    - detections = process_detections(result)    │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ d. Calculate metrics                             │  │
│   │    - create_comprehensive_results(...)          │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ e. Generate visualizations (if mode != "none")  │  │
│   │    - Per-image charts if needed                 │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ f. Save individual result                        │  │
│   │    - batch_dir/image_001/results.json           │  │
│   │    - batch_dir/image_001/visualizations/        │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ g. Update job progress                           │  │
│   │    - job_manager.add_job_result(...)            │  │
│   │    - job_manager.update_job_progress(i+1, N)    │  │
│   │    - Persist to disk                            │  │
│   └──────────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────────┐  │
│   │ h. Cleanup temp files                            │  │
│   └──────────────────────────────────────────────────┘  │
│                                                          │
│   [If error occurs]                                      │
│   ┌──────────────────────────────────────────────────┐  │
│   │ - Log error                                      │  │
│   │ - Add failed result to job                      │  │
│   │ - Continue to next image                        │  │
│   └──────────────────────────────────────────────────┘  │
└────┬─────────────────────────────────────────────────────┘
     │
     │ 3. All images processed
     ▼
┌─────────────────────────────────────┐
│   Generate Summary (if requested)   │
│   - Aggregate metrics               │
│   - Generate summary visualizations │
│   - Save to batch_dir/summary.json  │
└────┬────────────────────────────────┘
     │
     │ 4. Mark job completed
     │    job_manager.mark_job_completed(job_id)
     │    - status = "completed" or "partial"
     │    - completed_at = timestamp
     │    - Persist to disk
     ▼
┌─────────────────────────────────────┐
│   Background Task Complete          │
└─────────────────────────────────────┘
```

### 5.3 Client Polling Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     │ Loop until job complete:
     │
     │ 1. GET /api/batch-job/abc123
     ▼
┌─────────────────────────────────────┐
│   FastAPI Router                    │
│   @router.get("/api/batch-job/...")│
└────┬────────────────────────────────┘
     │
     │ 2. Fetch from manager
     │    job = job_manager.get_job(job_id)
     ▼
┌─────────────────────────────────────┐
│   BatchJobManager                   │
│   - Load from memory (fast)         │
│   - If not in memory, load from disk│
└────┬────────────────────────────────┘
     │
     │ 3. Return job status
     ▼
┌──────────┐  Receives:
│  Client  │  {
└──────────┘    "job_id": "abc123",
                "status": "processing",
                "progress": {
                  "current": 5,
                  "total": 10,
                  "percentage": 50.0
                },
                "results": [...]
              }
     │
     │ 4. Check status
     │    if status == "completed" or "partial":
     │        break
     │    else:
     │        sleep(2 seconds)
     │        continue loop
     ▼
┌──────────┐
│  Client  │  Job complete!
└──────────┘  Process final results
```

---

## 6. API Endpoint Specifications

### 6.1 Complete API Surface

```
EXISTING ENDPOINTS (Unchanged):
├── GET  /health
├── GET  /api/model-info
├── POST /api/detect
├── GET  /api/perturbations/info
├── POST /api/perturb
└── POST /api/detect-with-perturbation

NEW ENDPOINTS (Batch Processing):
├── POST /api/detect-batch          ← Main batch submission
└── GET  /api/batch-job/{job_id}    ← Status polling
```

### 6.2 POST /api/detect-batch - Detailed Specification

#### Request Format

**Content-Type:** `multipart/form-data`

**Form Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `files` | File[] | ✅ Yes | - | 1-300 image files |
| `score_thr` | string | ❌ No | `"0.3"` | Confidence threshold (0.0-1.0) |
| `perturbations` | string | ❌ No | `null` | JSON array (shared or per-image) |
| `visualization_mode` | string | ❌ No | `"none"` | `none`, `per_image`, `summary`, `both` |
| `save_json` | string | ❌ No | `"true"` | Save results to disk |
| `background_folder` | string | ❌ No | `null` | Background images path |

**Example cURL:**

```bash
curl -X POST "http://localhost:8000/api/detect-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "score_thr=0.3" \
  -F 'perturbations=[{"type":"defocus","degree":2}]' \
  -F "visualization_mode=summary"
```

**Example Python:**

```python
import requests

files = [
    ('files', ('img1.jpg', open('img1.jpg', 'rb'), 'image/jpeg')),
    ('files', ('img2.jpg', open('img2.jpg', 'rb'), 'image/jpeg')),
    ('files', ('img3.jpg', open('img3.jpg', 'rb'), 'image/jpeg')),
]

data = {
    'score_thr': '0.3',
    'perturbations': '[{"type":"defocus","degree":2}]',
    'visualization_mode': 'summary'
}

response = requests.post(
    'http://localhost:8000/api/detect-batch',
    files=files,
    data=data
)

job = response.json()
job_id = job['job_id']
print(f"Job started: {job_id}")
```

#### Response Format (202 Accepted)

```json
{
    "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "status": "queued",
    "message": "Batch processing started for 10 images",
    "total_images": 10,
    "status_endpoint": "/api/batch-job/3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "estimated_time_seconds": 30
}
```

#### Error Responses

| Status | Condition | Response Body |
|--------|-----------|---------------|
| 400 | No files provided | `{"detail": "At least one file is required"}` |
| 400 | Too many files (>300) | `{"detail": "Maximum 300 files allowed, got 350"}` |
| 400 | Invalid file type | `{"detail": "File xyz.pdf is not an image"}` |
| 400 | Invalid score_thr | `{"detail": "Invalid score_thr: Must be between 0 and 1"}` |
| 400 | Invalid perturbations JSON | `{"detail": "Invalid JSON in perturbations parameter"}` |
| 400 | Perturbation count mismatch | `{"detail": "Per-image perturbations count (5) must match file count (10)"}` |
| 500 | Server error | `{"success": false, "error": "Internal server error"}` |

### 6.3 GET /api/batch-job/{job_id} - Detailed Specification

#### Request Format

**Path Parameter:**
- `job_id` (string): UUID returned by `/api/detect-batch`

**Example cURL:**

```bash
curl "http://localhost:8000/api/batch-job/3fa85f64-5717-4562-b3fc-2c963f66afa6"
```

**Example Python (with polling):**

```python
import requests
import time

job_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
url = f"http://localhost:8000/api/batch-job/{job_id}"

while True:
    response = requests.get(url)
    job = response.json()
    
    status = job['status']
    progress = job['progress']
    
    print(f"Status: {status} - {progress['current']}/{progress['total']} "
          f"({progress['percentage']:.1f}%) - "
          f"Success: {progress['successful']}, Failed: {progress['failed']}")
    
    if status in ['completed', 'partial', 'failed']:
        print("\nJob finished!")
        print(f"Results saved to: {job.get('output_dir')}")
        break
    
    time.sleep(2)  # Poll every 2 seconds
```

#### Response Format - By Status

**Status: queued (initial state)**
```json
{
    "job_id": "abc123",
    "status": "queued",
    "created_at": "2024-11-23T14:30:22.123456",
    "updated_at": "2024-11-23T14:30:22.123456",
    "config": {
        "total_images": 10,
        "score_threshold": 0.3,
        "visualization_mode": "summary",
        "perturbation_mode": "shared",
        "filenames": ["img1.jpg", "img2.jpg", ...]
    },
    "progress": {
        "current": 0,
        "total": 10,
        "percentage": 0.0,
        "successful": 0,
        "failed": 0,
        "processing_times": []
    },
    "results": []
}
```

**Status: processing (in progress)**
```json
{
    "job_id": "abc123",
    "status": "processing",
    "created_at": "2024-11-23T14:30:22.123456",
    "updated_at": "2024-11-23T14:32:45.789012",
    "started_at": "2024-11-23T14:30:23.456789",
    "config": {...},
    "progress": {
        "current": 5,
        "total": 10,
        "percentage": 50.0,
        "successful": 4,
        "failed": 1,
        "processing_times": [2.3, 1.8, 3.1, 2.0, 2.5]
    },
    "results": [
        {
            "image": "img1.jpg",
            "status": "success",
            "processing_time": 2.3,
            "detections_count": 47,
            "average_confidence": 0.82,
            "result_path": "outputs/batch_20241123_143022/image_001/results.json"
        },
        {
            "image": "img2.jpg",
            "status": "success",
            "processing_time": 1.8,
            "detections_count": 52,
            "average_confidence": 0.79,
            "result_path": "outputs/batch_20241123_143022/image_002/results.json"
        },
        {
            "image": "img3.jpg",
            "status": "failed",
            "error": "Model inference failed: CUDA out of memory",
            "traceback": "Traceback (most recent call last):\n  File ..."
        },
        {
            "image": "img4.jpg",
            "status": "success",
            "processing_time": 2.0,
            "detections_count": 41,
            "average_confidence": 0.85,
            "result_path": "outputs/batch_20241123_143022/image_004/results.json"
        },
        {
            "image": "img5.jpg",
            "status": "success",
            "processing_time": 2.5,
            "detections_count": 38,
            "average_confidence": 0.77,
            "result_path": "outputs/batch_20241123_143022/image_005/results.json"
        }
    ]
}
```

**Status: completed (all successful)**
```json
{
    "job_id": "abc123",
    "status": "completed",
    "created_at": "2024-11-23T14:30:22.123456",
    "updated_at": "2024-11-23T14:35:56.789012",
    "started_at": "2024-11-23T14:30:23.456789",
    "completed_at": "2024-11-23T14:35:56.789012",
    "config": {...},
    "progress": {
        "current": 10,
        "total": 10,
        "percentage": 100.0,
        "successful": 10,
        "failed": 0,
        "processing_times": [2.3, 1.8, 3.1, 2.0, 2.5, 2.2, 1.9, 2.8, 2.1, 2.4]
    },
    "results": [
        // ... all 10 image results with status "success"
    ],
    "summary": {
        "total_images": 10,
        "successful_images": 10,
        "failed_images": 0,
        "total_detections": 456,
        "average_detections_per_image": 45.6,
        "average_confidence": 0.79,
        "confidence_std": 0.12,
        "class_distribution": {
            "paragraph": 152,
            "title": 84,
            "figure": 67,
            "table": 43,
            // ... more classes
        },
        "processing_time_total": 23.5,
        "processing_time_average": 2.35
    },
    "output_dir": "outputs/batch_20241123_143022"
}
```

**Status: partial (some failed)**
```json
{
    "job_id": "abc123",
    "status": "partial",
    "created_at": "2024-11-23T14:30:22.123456",
    "completed_at": "2024-11-23T14:35:56.789012",
    "config": {...},
    "progress": {
        "current": 10,
        "total": 10,
        "percentage": 100.0,
        "successful": 8,
        "failed": 2
    },
    "results": [
        // ... 8 successful, 2 failed
    ],
    "summary": {
        // ... summary based on successful images only
        "total_images": 10,
        "successful_images": 8,
        "failed_images": 2
    },
    "output_dir": "outputs/batch_20241123_143022"
}
```

**Status: failed (critical error or all failed)**
```json
{
    "job_id": "abc123",
    "status": "failed",
    "created_at": "2024-11-23T14:30:22.123456",
    "updated_at": "2024-11-23T14:31:45.123456",
    "error": "Critical error: Model failed to load",
    "progress": {
        "current": 0,
        "total": 10,
        "percentage": 0.0,
        "successful": 0,
        "failed": 0
    },
    "results": []
}
```

#### Error Responses

| Status | Condition | Response Body |
|--------|-----------|---------------|
| 404 | Job not found | `{"detail": "Job abc123 not found"}` |
| 500 | Server error | `{"success": false, "error": "..."}` |

---

## 7. Database/Storage Schema

### 7.1 In-Memory Storage

**Global Singleton Instance:**
```python
# In services/batch_job_manager.py
_job_manager_instance = None

def get_job_manager() -> BatchJobManager:
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = BatchJobManager(JOBS_DIR)
    return _job_manager_instance
```

**Thread-Safe Dictionary:**
```python
class BatchJobManager:
    def __init__(self, jobs_dir: Path):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
    
    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job_progress(self, job_id: str, current: int, total: int):
        with self._lock:
            # ... update logic
            self._persist_job(job_id)
```

### 7.2 File System Storage

**Directory Structure:**
```
outputs/
├── jobs/                                    # Job metadata
│   ├── job_abc123.json                      # Job state file
│   ├── job_def456.json
│   └── job_ghi789.json
│
├── batch_20241123_143022/                   # Batch results
│   ├── summary.json                         # Aggregate statistics
│   │
│   ├── image_001/                           # First image
│   │   ├── results.json                     # Full RoDLA results
│   │   └── visualizations/                  # (if per_image mode)
│   │       ├── class_distribution.png
│   │       ├── confidence_distribution.png
│   │       ├── spatial_heatmap.png
│   │       ├── confidence_by_class.png
│   │       ├── area_vs_confidence.png
│   │       ├── quadrant_distribution.png
│   │       ├── size_distribution.png
│   │       └── top_classes_confidence.png
│   │
│   ├── image_002/
│   │   ├── results.json
│   │   └── visualizations/
│   │
│   ├── image_003/
│   │   └── results.json                     # (failed image, no viz)
│   │
│   ├── ... (more images)
│   │
│   └── summary_visualizations/              # (if summary or both mode)
│       ├── combined_class_distribution.png
│       ├── combined_confidence_histogram.png
│       ├── average_confidence_by_class.png
│       ├── detection_count_per_image.png
│       ├── confidence_trend_across_images.png
│       ├── processing_time_per_image.png
│       ├── success_failure_summary.png
│       └── top_classes_across_batch.png
│
└── perturbations/                           # (existing, unchanged)
    └── ...
```

**Job File Schema (job_abc123.json):**
```json
{
    "job_id": "abc123",
    "status": "completed",
    "created_at": "2024-11-23T14:30:22.123456",
    "updated_at": "2024-11-23T14:35:56.789012",
    "started_at": "2024-11-23T14:30:23.456789",
    "completed_at": "2024-11-23T14:35:56.789012",
    
    "config": {
        "total_images": 10,
        "score_threshold": 0.3,
        "visualization_mode": "summary",
        "perturbations": [{"type": "defocus", "degree": 2}],
        "perturbation_mode": "shared",
        "save_json": true,
        "background_folder": null,
        "filenames": ["img1.jpg", "img2.jpg", ...]
    },
    
    "progress": {
        "current": 10,
        "total": 10,
        "percentage": 100.0,
        "successful": 10,
        "failed": 0,
        "processing_times": [2.3, 1.8, 3.1, 2.0, 2.5, 2.2, 1.9, 2.8, 2.1, 2.4]
    },
    
    "results": [
        {
            "image": "img1.jpg",
            "status": "success",
            "processing_time": 2.3,
            "detections_count": 47,
            "average_confidence": 0.82,
            "result_path": "outputs/batch_20241123_143022/image_001/results.json"
        },
        // ... more results
    ],
    
    "summary": {
        "total_images": 10,
        "successful_images": 10,
        "failed_images": 0,
        "total_detections": 456,
        "average_detections_per_image": 45.6,
        "average_confidence": 0.79,
        "confidence_std": 0.12,
        "class_distribution": {
            "paragraph": 152,
            "title": 84,
            // ... more classes
        },
        "processing_time_total": 23.5,
        "processing_time_average": 2.35
    },
    
    "output_dir": "outputs/batch_20241123_143022"
}
```

**Individual Image Result (image_001/results.json):**
```json
{
    // Standard RoDLA result format (unchanged from single-image API)
    "success": true,
    "timestamp": "2024-11-23T14:30:25.123456",
    "filename": "img1.jpg",
    "image_info": {...},
    "detection_config": {...},
    "core_results": {...},
    "rodla_metrics": {...},
    "spatial_analysis": {...},
    "class_analysis": {...},
    "confidence_analysis": {...},
    "robustness_indicators": {...},
    "layout_complexity": {...},
    "quality_metrics": {...},
    "interpretation": {...},
    "all_detections": [...]
}
```

**Batch Summary (summary.json):**
```json
{
    "batch_id": "batch_20241123_143022",
    "created_at": "2024-11-23T14:30:22.123456",
    "completed_at": "2024-11-23T14:35:56.789012",
    
    "statistics": {
        "total_images": 10,
        "successful_images": 10,
        "failed_images": 0,
        "total_detections": 456,
        "average_detections_per_image": 45.6,
        "average_confidence": 0.79,
        "confidence_std": 0.12,
        "min_confidence": 0.31,
        "max_confidence": 0.98,
        "processing_time_total": 23.5,
        "processing_time_average": 2.35,
        "processing_time_min": 1.8,
        "processing_time_max": 3.1
    },
    
    "class_distribution": {
        "paragraph": {
            "total_count": 152,
            "percentage": 33.3,
            "average_confidence": 0.81
        },
        "title": {
            "total_count": 84,
            "percentage": 18.4,
            "average_confidence": 0.87
        },
        // ... more classes
    },
    
    "per_image_summary": [
        {
            "image": "img1.jpg",
            "detections": 47,
            "avg_confidence": 0.82,
            "processing_time": 2.3,
            "status": "success"
        },
        // ... more images
    ],
    
    "visualizations_generated": ["combined_class_distribution", "..."]
}
```

### 7.3 Recovery on Restart

**Startup Logic:**
```python
class BatchJobManager:
    def _load_jobs_from_disk(self):
        """Load all existing jobs from disk on startup."""
        print(f"Loading jobs from {self.jobs_dir}...")
        
        job_files = list(self.jobs_dir.glob("job_*.json"))
        loaded_count = 0
        
        for job_file in job_files:
            try:
                with open(job_file, 'r') as f:
                    job = json.load(f)
                
                job_id = job["job_id"]
                
                # Mark incomplete jobs as failed
                if job["status"] in ["queued", "processing"]:
                    job["status"] = "failed"
                    job["error"] = "Server restarted during processing"
                    job["updated_at"] = datetime.now().isoformat()
                
                self._jobs[job_id] = job
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading {job_file}: {e}")
        
        print(f"Loaded {loaded_count} jobs from disk")
```

---

## 8. Error Handling Strategy

### 8.1 Error Categories and Responses

**1. Validation Errors (HTTP 400)**
```python
# File count validation
if len(files) > MAX_BATCH_SIZE:
    raise HTTPException(400, f"Maximum {MAX_BATCH_SIZE} files allowed")

# File type validation
for file in files:
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, f"{file.filename} is not an image")

# Perturbation validation
if pert_mode == "per_image" and len(perturbations) != len(files):
    raise HTTPException(400, "Perturbation count must match file count")
```

**2. Per-Image Errors (Continue Processing)**
```python
for i, image_path in enumerate(image_paths):
    try:
        # Process image
        result = process_single_image(image_path, config)
        
        job_manager.add_job_result(job_id, filename, {
            "status": "success",
            "data": result
        })
        
    except Exception as e:
        # Log but don't stop batch
        logger.error(f"Image {i} failed: {e}")
        
        job_manager.add_job_result(job_id, filename, {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    
    # Always update progress
    job_manager.update_job_progress(job_id, i+1, total)
```

**3. Critical Errors (Stop Batch)**
```python
try:
    # Batch processing loop
    for image in images:
        process_image(image)

except Exception as e:
    # Critical error - stop entire batch
    logger.critical(f"Critical error in batch {job_id}: {e}")
    
    job_manager.mark_job_failed(job_id, f"Critical error: {str(e)}")
    
    # Optionally notify client (future: webhook)
```

### 8.2 Retry Logic (Future Enhancement)

**Per-Image Retry (Not in MVP):**
```python
MAX_RETRIES = 3

for attempt in range(MAX_RETRIES):
    try:
        result = process_image(image)
        break  # Success
    except Exception as e:
        if attempt == MAX_RETRIES - 1:
            # Final attempt failed
            mark_failed(image, error=e)
        else:
            # Retry with backoff
            time.sleep(2 ** attempt)
```

### 8.3 Resource Cleanup

**Temp File Cleanup:**
```python
def process_batch_job(job_id, temp_paths, config, job_manager):
    temp_files_to_cleanup = temp_paths.copy()
    
    try:
        # Process images
        for temp_path in temp_paths:
            try:
                # ... processing ...
                temp_files_to_cleanup.remove(temp_path)
                os.unlink(temp_path)
            except Exception:
                pass  # Cleanup in finally
    
    finally:
        # Ensure all temp files deleted
        for temp_path in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_path}: {e}")
```

**GPU Memory Cleanup:**
```python
import gc
import torch

def process_image(image_path):
    try:
        # ... inference ...
        return result
    finally:
        # Clean up GPU memory after each image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
```

---

## 9. Implementation Checklist

### Phase 1: Core Infrastructure (2-3 hours)

- [ ] **Create `services/batch_job_manager.py`**
  - [ ] Implement `BatchJobManager` class
  - [ ] Implement `create_job()` method
  - [ ] Implement `get_job()` method
  - [ ] Implement `update_job_progress()` method
  - [ ] Implement `add_job_result()` method
  - [ ] Implement `mark_job_completed()` method
  - [ ] Implement `mark_job_failed()` method
  - [ ] Implement `_persist_job()` method (JSON save)
  - [ ] Implement `_load_jobs_from_disk()` method
  - [ ] Add thread safety (locks)
  - [ ] Test in isolation (unit tests)

- [ ] **Update `config/settings.py`**
  - [ ] Add `JOBS_DIR` constant
  - [ ] Add `MAX_BATCH_SIZE = 300`
  - [ ] Add `MIN_BATCH_SIZE = 1`
  - [ ] Add `JOB_RETENTION_HOURS = 48`
  - [ ] Add `BATCH_OUTPUT_PREFIX = "batch_"`
  - [ ] Add `VISUALIZATION_MODES` list
  - [ ] Add `ESTIMATED_TIME_PER_IMAGE = 3.0`
  - [ ] Create `JOBS_DIR` on import

- [ ] **Update `api/schemas.py`**
  - [ ] Add `BatchJobConfig` model
  - [ ] Add `BatchJobProgress` model
  - [ ] Add `BatchImageResult` model
  - [ ] Add `BatchJobSummary` model
  - [ ] Add `BatchJobStatus` model
  - [ ] Add `BatchJobCreateResponse` model

### Phase 2: Batch Processing Logic (3-4 hours)

- [ ] **Update `services/processing.py`**
  - [ ] Add `process_batch_job()` async function
  - [ ] Add `get_perturbations_for_image()` helper
  - [ ] Add `save_individual_result()` helper
  - [ ] Add `save_temp_file()` helper
  - [ ] Add `cleanup_temp_files()` helper
  - [ ] Add `generate_batch_summary()` function
  - [ ] Add `generate_summary_visualizations()` function
  - [ ] Add `save_summary()` function
  - [ ] Test batch processing with 3-5 images

### Phase 3: API Endpoints (2-3 hours)

- [ ] **Update `api/routes.py`**
  - [ ] Add `POST /api/detect-batch` endpoint
    - [ ] Validate file count (1-300)
    - [ ] Validate file types
    - [ ] Parse score_thr
    - [ ] Parse and validate perturbations
    - [ ] Validate visualization_mode
    - [ ] Create job config
    - [ ] Create job in manager
    - [ ] Save uploaded files to temp
    - [ ] Launch BackgroundTask
    - [ ] Return 202 response with job_id
  - [ ] Add `GET /api/batch-job/{job_id}` endpoint
    - [ ] Get job from manager
    - [ ] Return 404 if not found
    - [ ] Return current job status
  - [ ] Add `get_job_manager()` dependency function
  - [ ] Test endpoints with Postman/cURL

### Phase 4: Visualization Enhancements (2-3 hours)

- [ ] **Update `services/visualization.py`**
  - [ ] Add `generate_summary_visualizations()` function
  - [ ] Implement combined_class_distribution chart
  - [ ] Implement combined_confidence_histogram chart
  - [ ] Implement detection_count_per_image chart
  - [ ] Implement confidence_trend_across_images chart
  - [ ] Implement processing_time_per_image chart
  - [ ] Implement success_failure_summary chart
  - [ ] Implement top_classes_across_batch chart
  - [ ] Implement average_confidence_by_class chart
  - [ ] Test with sample batch results

### Phase 5: Integration & Testing (2-3 hours)

- [ ] **Integration Testing**
  - [ ] Test single image batch (edge case)
  - [ ] Test small batch (5 images)
  - [ ] Test medium batch (50 images)
  - [ ] Test large batch (200 images)
  - [ ] Test maximum batch (300 images)
  - [ ] Test with shared perturbations
  - [ ] Test with per-image perturbations
  - [ ] Test visualization_mode="none"
  - [ ] Test visualization_mode="per_image"
  - [ ] Test visualization_mode="summary"
  - [ ] Test visualization_mode="both"
  - [ ] Test error scenarios (invalid files, OOM, etc.)
  - [ ] Test job recovery after restart

- [ ] **Performance Testing**
  - [ ] Measure processing time per image
  - [ ] Monitor GPU memory usage
  - [ ] Monitor disk space usage
  - [ ] Test concurrent batch submissions (if needed)

- [ ] **Error Testing**
  - [ ] Test with non-image files
  - [ ] Test with corrupted images
  - [ ] Test with too many files (>300)
  - [ ] Test with invalid perturbations
  - [ ] Test with mismatched perturbation counts
  - [ ] Test GPU OOM recovery
  - [ ] Test disk full scenario

### Phase 6: Documentation (1-2 hours)

- [ ] **Update README.md**
  - [ ] Add batch processing section
  - [ ] Add API endpoint documentation
  - [ ] Add usage examples (cURL + Python)
  - [ ] Add visualization mode explanation
  - [ ] Add error handling documentation
  - [ ] Add performance benchmarks

- [ ] **Create API Examples**
  - [ ] Python client example
  - [ ] cURL examples
  - [ ] Postman collection (optional)

### Phase 7: Backend Updates (1 hour)

- [ ] **Update `backend.py`**
  - [ ] Initialize job manager on startup
  - [ ] Load existing jobs from disk
  - [ ] Log startup messages
  - [ ] No other changes needed (routes auto-included)

---

## 10. Testing Strategy

### 10.1 Unit Tests

**Test `BatchJobManager`:**
```python
# tests/test_services/test_batch_job_manager.py

def test_create_job():
    manager = BatchJobManager(tmp_path / "jobs")
    config = {"total_images": 5}
    
    job_id = manager.create_job(config)
    
    assert job_id is not None
    assert len(job_id) == 36  # UUID4 length
    
    job = manager.get_job(job_id)
    assert job["status"] == "queued"
    assert job["config"] == config


def test_update_progress():
    manager = BatchJobManager(tmp_path / "jobs")
    job_id = manager.create_job({"total_images": 10})
    
    manager.update_job_progress(job_id, 5, 10)
    
    job = manager.get_job(job_id)
    assert job["progress"]["current"] == 5
    assert job["progress"]["percentage"] == 50.0


def test_persist_and_load():
    jobs_dir = tmp_path / "jobs"
    manager1 = BatchJobManager(jobs_dir)
    job_id = manager1.create_job({"total_images": 3})
    
    # Create new manager (simulates restart)
    manager2 = BatchJobManager(jobs_dir)
    job = manager2.get_job(job_id)
    
    assert job is not None
    assert job["status"] == "failed"  # Marked as failed on restart
```

### 10.2 Integration Tests

**Test Full Batch Pipeline:**
```python
# tests/test_integration/test_batch_pipeline.py

@pytest.mark.slow
def test_small_batch_end_to_end(test_client, sample_images):
    # Submit batch
    files = [
        ('files', ('img1.jpg', open(sample_images[0], 'rb'), 'image/jpeg')),
        ('files', ('img2.jpg', open(sample_images[1], 'rb'), 'image/jpeg')),
        ('files', ('img3.jpg', open(sample_images[2], 'rb'), 'image/jpeg')),
    ]
    
    response = test_client.post(
        '/api/detect-batch',
        files=files,
        data={'score_thr': '0.3', 'visualization_mode': 'summary'}
    )
    
    assert response.status_code == 202
    job_id = response.json()['job_id']
    
    # Poll until complete
    max_polls = 30
    for _ in range(max_polls):
        status_response = test_client.get(f'/api/batch-job/{job_id}')
        job = status_response.json()
        
        if job['status'] in ['completed', 'partial', 'failed']:
            break
        
        time.sleep(2)
    
    # Verify completion
    assert job['status'] == 'completed'
    assert job['progress']['successful'] == 3
    assert len(job['results']) == 3
    assert job['summary'] is not None


@pytest.mark.slow
def test_batch_with_perturbations(test_client, sample_images):
    files = [
        ('files', ('img1.jpg', open(sample_images[0], 'rb'), 'image/jpeg')),
        ('files', ('img2.jpg', open(sample_images[1], 'rb'), 'image/jpeg')),
    ]
    
    perturbations = [
        [{"type": "defocus", "degree": 2}],  # For img1
        [{"type": "speckle", "degree": 1}],  # For img2
    ]
    
    response = test_client.post(
        '/api/detect-batch',
        files=files,
        data={
            'perturbations': json.dumps(perturbations),
            'visualization_mode': 'none'
        }
    )
    
    assert response.status_code == 202
    job_id = response.json()['job_id']
    
    # Wait and verify
    # ... (similar polling logic)
```

### 10.3 Load Tests (Optional)

**Test Multiple Concurrent Batches:**
```python
import concurrent.futures

def submit_batch(client, num_images):
    files = [...]  # Create files
    response = client.post('/api/detect-batch', files=files)
    return response.json()['job_id']

def test_concurrent_batches():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(submit_batch, client, 10)
            for _ in range(5)
        ]
        
        job_ids = [f.result() for f in futures]
        
        # Verify all jobs eventually complete
        # ...
```

---

## 11. Migration Path

###