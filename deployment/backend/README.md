# RoDLA Document Layout Analysis API

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![CVPR](https://img.shields.io/badge/CVPR-2024-purple.svg)

**A Production-Ready API for Robust Document Layout Analysis**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Architecture](#-architecture) • [Metrics](#-metrics-system)

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
10. [Metrics System](#-metrics-system)
11. [Visualization Engine](#-visualization-engine)
12. [Services Layer](#-services-layer)
13. [Utilities Reference](#-utilities-reference)
14. [Error Handling](#-error-handling)
15. [Performance Optimization](#-performance-optimization)
16. [Security Considerations](#-security-considerations)
17. [Testing](#-testing)
18. [Deployment](#-deployment)
19. [Troubleshooting](#-troubleshooting)
20. [Contributing](#-contributing)
21. [Citation](#-citation)
22. [License](#-license)

---

## 🎯 Overview

### What is RoDLA?

RoDLA (Robust Document Layout Analysis) is a state-of-the-art deep learning model for detecting and classifying layout elements in document images. Published at **CVPR 2024**, it focuses on robustness to various perturbations including noise, blur, and geometric distortions.

### What is this API?

This repository provides a **production-ready FastAPI wrapper** around the RoDLA model, featuring:

- RESTful API endpoints for document analysis
- Comprehensive metrics calculation (20+ metrics)
- Automated visualization generation (8 chart types)
- Robustness assessment based on the RoDLA paper
- Human-readable interpretation of results
- Modular, maintainable code architecture

### Key Statistics

| Metric | Value |
|--------|-------|
| Clean mAP (M6Doc) | 70.0% |
| Perturbed Average mAP | 61.7% |
| mRD Score | 147.6 |
| Max Detections/Image | 300 |
| Supported Classes | 74 (M6Doc) |

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

```
# Core Framework
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6

# ML/Deep Learning
torch>=2.0.0
mmdet>=3.0.0
mmcv>=2.0.0

# Data Processing
numpy>=1.24.0
pillow>=9.5.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
pydantic>=2.0.0
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/rodla-api.git
cd rodla-api
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
```

### Step 4: Install MMDetection

```bash
pip install -U openmim
mim install mmengine
mim install mmcv>=2.0.0
mim install mmdet>=3.0.0
```

### Step 5: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Download Model Weights

```bash
# Download from official source
wget https://path-to-weights/rodla_internimage_xl_m6doc.pth -O weights/rodla_internimage_xl_m6doc.pth
```

### Step 7: Configure Paths

Edit `config/settings.py`:

```python
REPO_ROOT = Path("/path/to/your/RoDLA")
MODEL_CONFIG = REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"
MODEL_WEIGHTS = REPO_ROOT / "rodla_internimage_xl_m6doc.pth"
```

---

## ⚡ Quick Start

### Starting the Server

```bash
# Development mode
python backend.py

# Production mode with uvicorn
uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 1
```

### Making Your First Request

```bash
# Using curl
curl -X POST "http://localhost:8000/api/detect" \
  -H "accept: application/json" \
  -F "file=@document.jpg" \
  -F "score_thr=0.3"

# Get model information
curl http://localhost:8000/api/model-info
```

### Python Client Example

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
```

---

## 📁 Project Structure

```
deployment/
├── backend.py                 # 🚀 Main FastAPI application entry point
├── requirements.txt           # 📦 Python dependencies
├── README.md                  # 📖 This documentation
│
├── config/                    # ⚙️ Configuration Layer
│   ├── __init__.py           #    Package initializer
│   └── settings.py           #    All configuration constants
│
├── core/                      # 🧠 Core Application Layer
│   ├── __init__.py           #    Package initializer
│   ├── model_loader.py       #    Singleton model management
│   └── dependencies.py       #    FastAPI dependency injection
│
├── api/                       # 🌐 API Layer
│   ├── __init__.py           #    Package initializer
│   ├── routes.py             #    API endpoint definitions
│   └── schemas.py            #    Pydantic request/response models
│
├── services/                  # 🔧 Business Logic Layer
│   ├── __init__.py           #    Package initializer
│   ├── detection.py          #    Core detection logic
│   ├── processing.py         #    Result aggregation
│   ├── visualization.py      #    Chart generation (350+ lines)
│   └── interpretation.py     #    Human-readable insights
│
├── utils/                     # 🛠️ Utility Layer
│   ├── __init__.py           #    Package initializer
│   ├── helpers.py            #    General helper functions
│   ├── serialization.py      #    JSON conversion utilities
│   └── metrics/              #    Metrics calculation modules
│       ├── __init__.py       #    Metrics package initializer
│       ├── core.py           #    Core detection metrics
│       ├── rodla.py          #    RoDLA-specific metrics
│       ├── spatial.py        #    Spatial distribution analysis
│       └── quality.py        #    Quality & complexity metrics
│
└── outputs/                   # 📤 Output Directory
    ├── *.json                #    Detection results
    └── *.png                 #    Visualization images
```

### File Count Summary

| Layer | Files | Purpose |
|-------|-------|---------|
| Config | 2 | Configuration management |
| Core | 3 | Model and dependency management |
| API | 3 | HTTP endpoints and schemas |
| Services | 5 | Business logic implementation |
| Utils | 7 | Helper functions and metrics |
| **Total** | **21** | Complete modular architecture |

---

## 🏗️ Architecture Deep Dive

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                           │
│              (Web Browser / API Clients)                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP Requests
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       API LAYER                             │
│                    api/routes.py                            │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ GET /model-info │  │ POST /api/detect                │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ Validated Requests
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    SERVICES LAYER                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐│
│  │ detection.py │ │processing.py │ │ visualization.py     ││
│  │              │ │              │ │                      ││
│  │ • Inference  │ │ • Aggregate  │ │ • 8 Chart Types      ││
│  │ • Processing │ │ • Save JSON  │ │ • Base64 Encoding    ││
│  └──────────────┘ └──────────────┘ └──────────────────────┘│
│  ┌──────────────────────────────────────────────────────┐  │
│  │               interpretation.py                       │  │
│  │         • Human-readable insights                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ Data Processing
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    UTILITIES LAYER                          │
│  ┌────────────────────────────────────────────────────────┐│
│  │                  utils/metrics/                        ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐  ││
│  │  │ core.py │ │rodla.py │ │spatial. │ │ quality.py  │  ││
│  │  │         │ │         │ │  py     │ │             │  ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘  ││
│  └────────────────────────────────────────────────────────┘│
│  ┌──────────────┐ ┌────────────────────────────────────┐   │
│  │ helpers.py   │ │ serialization.py                   │   │
│  └──────────────┘ └────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │ Model Operations
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE LAYER                             │
│  ┌─────────────────────────┐ ┌───────────────────────────┐ │
│  │    model_loader.py      │ │    dependencies.py        │ │
│  │                         │ │                           │ │
│  │ • Singleton Pattern     │ │ • FastAPI DI              │ │
│  │ • GPU Management        │ │ • Model Injection         │ │
│  │ • Lazy Loading          │ │                           │ │
│  └─────────────────────────┘ └───────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │ Configuration
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONFIG LAYER                             │
│                   config/settings.py                        │
│  • Paths  • Constants  • Baseline Metrics  • Thresholds    │
└─────────────────────────────────────────────────────────────┘
```

### Design Patterns Used

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Singleton** | `model_loader.py` | Single model instance |
| **Factory** | `visualization.py` | Create multiple chart types |
| **Dependency Injection** | `dependencies.py` | Inject model into routes |
| **Repository** | `processing.py` | Abstract data persistence |
| **Facade** | `routes.py` | Simplify complex subsystems |
| **Strategy** | `metrics/` | Interchangeable metric algorithms |

### Data Flow Diagram

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Image   │───▶│  Upload  │───▶│  Temp    │───▶│  Model   │
│  File    │    │  Handler │    │  File    │    │ Inference│
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                     │
     ┌───────────────────────────────────────────────┘
     ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Raw     │───▶│ Process  │───▶│ Calculate│───▶│ Generate │
│  Results │    │Detections│    │ Metrics  │    │  Viz     │
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                     │
     ┌───────────────────────────────────────────────┘
     ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Generate │───▶│ Assemble │───▶│  JSON    │
│ Interp.  │    │ Response │    │ Response │
└──────────┘    └──────────┘    └──────────┘
```

---

## ⚙️ Configuration

### config/settings.py

This file centralizes all configuration parameters.

```python
"""
Configuration Settings Module
=============================
All application constants and configuration in one place.
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
# API CONFIGURATION
# =============================================================================

# CORS settings
CORS_ORIGINS = ["*"]  # Restrict in production
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# API metadata
API_TITLE = "RoDLA Object Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Production-ready API for Robust Document Layout Analysis"

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
RODLA_DEFAULT_THRESHOLD=0.3
RODLA_API_HOST=0.0.0.0
RODLA_API_PORT=8000
```

---

## 🌐 API Reference

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/model-info` | Get model metadata |
| POST | `/api/detect` | Analyze document image |
| GET | `/health` | Health check (if implemented) |
| GET | `/docs` | Swagger UI documentation |
| GET | `/redoc` | ReDoc documentation |

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
        // ... additional classes
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

#### Error Responses

| Status | Description |
|--------|-------------|
| 500 | Model not loaded |

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
| `file` | File | Required | Image file (JPEG, PNG, etc.) |
| `score_thr` | string | "0.3" | Confidence threshold (0.0-1.0) |
| `return_image` | string | "false" | Return annotated image instead of JSON |
| `save_json` | string | "true" | Save results to disk |
| `generate_visualizations` | string | "true" | Generate visualization charts |

#### Response (JSON Mode)

```json
{
    "success": true,
    "timestamp": "2024-01-15T10:30:45.123456",
    "filename": "document.jpg",
    
    "image_info": {
        "width": 2480,
        "height": 3508,
        "aspect_ratio": 0.707,
        "total_pixels": 8699840
    },
    
    "detection_config": {
        "score_threshold": 0.3,
        "model": "RoDLA InternImage-XL",
        "framework": "DINO with Robustness Enhancement",
        "max_detections": 300
    },
    
    "core_results": {
        "summary": {
            "total_detections": 47,
            "unique_classes": 12,
            "average_confidence": 0.7823,
            "median_confidence": 0.8156,
            "min_confidence": 0.3012,
            "max_confidence": 0.9876,
            "coverage_percentage": 68.45,
            "average_detection_area": 126543.21
        },
        "detections": [/* top 20 detections */]
    },
    
    "rodla_metrics": {
        "note": "Estimated metrics...",
        "estimated_mPE": 18.45,
        "estimated_mRD": 87.32,
        "confidence_std": 0.1234,
        "confidence_range": 0.6864,
        "robustness_score": 56.34,
        "interpretation": {
            "mPE_level": "low",
            "mRD_level": "excellent",
            "overall_robustness": "medium"
        }
    },
    
    "spatial_analysis": {
        "horizontal_distribution": {...},
        "vertical_distribution": {...},
        "quadrant_distribution": {...},
        "size_distribution": {...},
        "density_metrics": {...}
    },
    
    "class_analysis": {
        "paragraph": {
            "count": 15,
            "percentage": 31.91,
            "confidence_stats": {...},
            "area_stats": {...},
            "aspect_ratio_stats": {...}
        },
        // ... other classes
    },
    
    "confidence_analysis": {
        "distribution": {...},
        "binned_distribution": {...},
        "percentages": {...},
        "entropy": 2.3456
    },
    
    "robustness_indicators": {
        "stability_score": 87.65,
        "coefficient_of_variation": 0.1234,
        "high_confidence_ratio": 0.7234,
        "prediction_consistency": "high",
        "model_certainty": "medium",
        "robustness_rating": {
            "rating": "good",
            "score": 72.34
        }
    },
    
    "layout_complexity": {
        "class_diversity": 12,
        "total_elements": 47,
        "detection_density": 5.41,
        "average_element_distance": 234.56,
        "complexity_score": 58.23,
        "complexity_level": "moderate",
        "layout_characteristics": {
            "is_dense": true,
            "is_diverse": true,
            "is_structured": false
        }
    },
    
    "quality_metrics": {
        "overlap_analysis": {...},
        "size_consistency": {...},
        "detection_quality_score": 82.45
    },
    
    "visualizations": {
        "class_distribution": "data:image/png;base64,...",
        "confidence_distribution": "data:image/png;base64,...",
        "spatial_heatmap": "data:image/png;base64,...",
        "confidence_by_class": "data:image/png;base64,...",
        "area_vs_confidence": "data:image/png;base64,...",
        "quadrant_distribution": "data:image/png;base64,...",
        "size_distribution": "data:image/png;base64,...",
        "top_classes_confidence": "data:image/png;base64,..."
    },
    
    "interpretation": {
        "overview": "Document Analysis Summary...",
        "top_elements": "The most common elements are...",
        "rodla_analysis": "RoDLA Robustness Analysis...",
        "layout_complexity": "Layout Complexity...",
        "key_findings": [...],
        "perturbation_assessment": "...",
        "recommendations": [...],
        "confidence_summary": {...}
    },
    
    "all_detections": [/* complete detection list */]
}
```

#### Response (Image Mode)

When `return_image=true`, returns the annotated image directly:

```http
HTTP/1.1 200 OK
Content-Type: image/jpeg
Content-Disposition: attachment; filename="annotated_document.jpg"

<binary image data>
```

#### Error Responses

| Status | Description |
|--------|-------------|
| 400 | Invalid file type (not an image) |
| 500 | Model inference failed |
| 500 | Visualization generation failed |

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

| Metric | Type | Description |
|--------|------|-------------|
| `total_detections` | int | Number of detected elements |
| `unique_classes` | int | Number of distinct element types |
| `average_confidence` | float | Mean confidence score |
| `median_confidence` | float | Median confidence score |
| `min_confidence` | float | Lowest confidence |
| `max_confidence` | float | Highest confidence |
| `coverage_percentage` | float | % of image covered by detections |
| `average_detection_area` | float | Mean area per detection |

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
    }
}
```

#### `calculate_confidence_metrics(detections)`

Detailed confidence distribution analysis.

| Component | Description |
|-----------|-------------|
| `distribution` | Statistical measures (mean, median, std, quartiles) |
| `binned_distribution` | Count per confidence range |
| `percentages` | Percentage per confidence range |
| `entropy` | Shannon entropy of distribution |

**Confidence Bins:**
- Very High: 0.9 - 1.0
- High: 0.8 - 0.9
- Medium: 0.6 - 0.8
- Low: 0.4 - 0.6
- Very Low: 0.0 - 0.4

---

### RoDLA Metrics (utils/metrics/rodla.py)

These metrics are specific to the RoDLA paper's robustness evaluation framework.

#### `calculate_rodla_metrics(detections, core_metrics)`

Estimates perturbation effects and robustness degradation.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `estimated_mPE` | `(conf_std × 100) + (conf_range × 50)` | Mean Perturbation Effect |
| `estimated_mRD` | `(degradation / mPE) × 100` | Mean Robustness Degradation |
| `robustness_score` | `(1 - mRD/200) × 100` | Overall robustness (0-100) |

**mPE Interpretation:**
```
low:    mPE < 20   → Minimal perturbation effect
medium: 20 ≤ mPE < 40 → Moderate perturbation
high:   mPE ≥ 40   → Significant perturbation
```

**mRD Interpretation:**
```
excellent:        mRD < 100  → Highly robust
good:             100 ≤ mRD < 150 → Acceptable robustness
needs_improvement: mRD ≥ 150 → Robustness concerns
```

#### `calculate_robustness_indicators(detections, core_metrics)`

Stability and consistency metrics.

```python
{
    "stability_score": 87.65,        # (1 - CV) × 100
    "coefficient_of_variation": 0.12, # std / mean
    "high_confidence_ratio": 0.72,   # % with conf ≥ 0.8
    "prediction_consistency": "high", # Based on CV
    "model_certainty": "medium",      # Based on avg conf
    "robustness_rating": {
        "rating": "good",             # excellent/good/fair/poor
        "score": 72.34                # Composite score
    }
}
```

**Robustness Rating Formula:**
```
score = (avg_conf × 40) + ((1 - CV) × 30) + (high_conf_ratio × 30)

Rating:
- excellent: score ≥ 80
- good:      60 ≤ score < 80
- fair:      40 ≤ score < 60
- poor:      score < 40
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
    "top_third": 8,           # Count in top 33%
    "middle_third": 22,       # Count in middle 33%
    "bottom_third": 17        # Count in bottom 33%
}
```

##### Quadrant Distribution
```
Document divided into 4 quadrants:
┌─────────┬─────────┐
│   Q1    │   Q2    │
│(top-L)  │(top-R)  │
├─────────┼─────────┤
│   Q3    │   Q4    │
│(bot-L)  │(bot-R)  │
└─────────┴─────────┘
```

##### Size Distribution
| Category | Threshold | Description |
|----------|-----------|-------------|
| tiny | < 0.5% of image | Very small elements |
| small | 0.5% - 2% | Small elements |
| medium | 2% - 10% | Medium elements |
| large | ≥ 10% | Large elements |

##### Density Metrics
```python
{
    "average_nearest_neighbor_distance": 234.56,  # pixels
    "spatial_clustering_score": 0.67              # 0-1, higher = more clustered
}
```

---

### Quality Metrics (utils/metrics/quality.py)

#### `calculate_layout_complexity(detections, img_width, img_height)`

Quantifies document structure complexity.

**Complexity Score Formula:**
```
score = (class_diversity / 20) × 30      # Max 20 classes
      + min(detections / 50, 1) × 30     # Detection count
      + min(density / 10, 1) × 20        # Spatial density
      + (1 - min(avg_dist / 500, 1)) × 20 # Clustering
```

**Complexity Levels:**
| Level | Score Range | Description |
|-------|-------------|-------------|
| simple | < 30 | Basic document layout |
| moderate | 30 - 60 | Average complexity |
| complex | ≥ 60 | Complex multi-element layout |

**Layout Characteristics:**
```python
{
    "is_dense": True,       # density > 5 elements/megapixel
    "is_diverse": True,     # unique_classes ≥ 10
    "is_structured": False  # avg_distance < 200 pixels
}
```

#### `calculate_quality_metrics(detections, img_width, img_height)`

Detection quality assessment.

##### Overlap Analysis
```python
{
    "total_overlapping_pairs": 5,    # Number of overlapping detection pairs
    "overlap_percentage": 10.64,      # % of detections with overlaps
    "average_iou": 0.1234             # Mean IoU of overlapping pairs
}
```

##### Size Consistency
```python
{
    "coefficient_of_variation": 0.876,  # std/mean of areas
    "consistency_level": "medium"        # high (<0.5), medium (0.5-1), low (>1)
}
```

##### Detection Quality Score
```
score = (1 - min(overlap_% / 100, 1)) × 50 + (1 - min(size_cv, 1)) × 50
```

---

## 📈 Visualization Engine

### services/visualization.py

The visualization engine generates 8 distinct chart types, each providing unique insights into the detection results.

### Chart Types

#### 1. Class Distribution Bar Chart
```
Purpose: Show count of detections per class
Type: Vertical bar chart
Features:
  - Sorted by count (descending)
  - Value labels on bars
  - Rotated x-axis labels for readability
  - Grid lines for easy reading
```

#### 2. Confidence Distribution Histogram
```
Purpose: Show distribution of confidence scores
Type: Histogram with 20 bins
Features:
  - Mean line (red dashed)
  - Median line (orange dashed)
  - Legend with exact values
  - Grid lines
```

#### 3. Spatial Distribution Heatmap
```
Purpose: Visualize where detections are concentrated
Type: 2D histogram heatmap
Features:
  - YlOrRd colormap (yellow to red)
  - Colorbar showing density
  - Axes showing pixel coordinates
```

#### 4. Confidence by Class Box Plot
```
Purpose: Compare confidence distributions across classes
Type: Box plot
Features:
  - Top 10 classes by count
  - Sample sizes in labels
  - Median, quartiles, outliers
  - Light blue boxes
```

#### 5. Area vs Confidence Scatter Plot
```
Purpose: Examine relationship between size and confidence
Type: Scatter plot
Features:
  - Color-coded by confidence (viridis)
  - Colorbar showing scale
  - Grid for reading values
```

#### 6. Quadrant Distribution Pie Chart
```
Purpose: Show spatial distribution by quadrant
Type: Pie chart
Features:
  - 4 segments (Q1-Q4)
  - Percentage labels
  - Element counts in labels
  - Distinct colors per quadrant
```

#### 7. Size Distribution Bar Chart
```
Purpose: Show distribution of detection sizes
Type: Vertical bar chart
Features:
  - 4 categories (tiny, small, medium, large)
  - Distinct color per category
  - Value labels on bars
```

#### 8. Top Classes by Average Confidence
```
Purpose: Identify most confidently detected classes
Type: Horizontal bar chart
Features:
  - Top 15 classes
  - Sorted by confidence
  - Value labels
  - Coral color scheme
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
    
    Returns:
        Dictionary with base64-encoded PNG images
    """
    visualizations = {}
    
    # Each visualization wrapped in try-except for isolation
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        # ... chart generation code ...
        visualizations['chart_name'] = fig_to_base64(fig)
        plt.close(fig)  # Prevent memory leaks
    except Exception as e:
        print(f"Error generating chart: {e}")
    
    return visualizations
```

### Base64 Encoding

```python
def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 data URI."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return f"data:image/png;base64,{image_base64}"
```

### Usage in HTML

```html
<img src="{{ visualizations.class_distribution }}" alt="Class Distribution">
```

---

## 🔧 Services Layer

### services/detection.py

Core detection logic and result processing.

#### `process_detections(result, score_thr=0.3)`

Converts raw model output to structured format.

**Input:** Raw MMDetection result (list of arrays per class)

**Output:** List of detection dictionaries

```python
[
    {
        "class_id": 0,
        "class_name": "paragraph",
        "bbox": {
            "x1": 100.5, "y1": 200.3,
            "x2": 500.8, "y2": 350.2,
            "width": 400.3, "height": 149.9,
            "center_x": 300.65, "center_y": 275.25
        },
        "confidence": 0.9234,
        "area": 60005.0,
        "aspect_ratio": 2.67
    },
    // ... more detections
]
```

**Processing Steps:**
1. Iterate through class results
2. Filter by confidence threshold
3. Extract coordinates and calculate derived values
4. Sort by confidence (descending)

---

### services/processing.py

Result aggregation and persistence.

#### `aggregate_results(...)`

Assembles the complete response object.

```python
def aggregate_results(
    detections: List[dict],
    core_metrics: dict,
    rodla_metrics: dict,
    spatial_metrics: dict,
    class_metrics: dict,
    confidence_metrics: dict,
    robustness_indicators: dict,
    layout_complexity: dict,
    quality_metrics: dict,
    visualizations: dict,
    interpretation: dict,
    file_info: dict,
    config: dict
) -> dict:
    """Combine all analysis results into final response."""
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        # ... all components ...
    }
```

#### `save_results(results, filename, output_dir)`

Persists results to disk.

```python
def save_results(results: dict, filename: str, output_dir: Path) -> Path:
    """
    Save results as JSON file.
    
    - Removes visualizations to reduce file size
    - Converts numpy types to Python native
    - Saves visualizations as separate PNG files
    """
    json_path = output_dir / f"rodla_results_{filename}.json"
    # ... save logic ...
    return json_path
```

---

### services/interpretation.py

Human-readable insight generation.

#### `generate_comprehensive_interpretation(...)`

Creates natural language analysis of results.

**Output Sections:**

| Section | Description |
|---------|-------------|
| `overview` | High-level summary paragraph |
| `top_elements` | Description of most common elements |
| `rodla_analysis` | Robustness assessment summary |
| `layout_complexity` | Complexity analysis text |
| `key_findings` | List of important observations |
| `perturbation_assessment` | Perturbation effect analysis |
| `recommendations` | Actionable suggestions |
| `confidence_summary` | Confidence level summary |

**Example Output:**

```python
{
    "overview": """Document Analysis Summary:
Detected 47 layout elements across 12 different classes.
The model achieved an average confidence of 78.2%, indicating
medium certainty in predictions. The detected elements cover
68.5% of the document area.""",

    "key_findings": [
        "✓ Excellent detection confidence - model is highly certain",
        "✓ High document coverage - most of the page contains elements",
        "ℹ Complex document structure with diverse element types"
    ],

    "recommendations": [
        "No specific recommendations - detection quality is good"
    ]
}
```

---

## 🛠️ Utilities Reference

### utils/helpers.py

General-purpose helper functions.

#### Mathematical Functions

| Function | Purpose | Formula |
|----------|---------|---------|
| `calculate_skewness(data)` | Distribution asymmetry | `mean(((x - μ) / σ)³)` |
| `calculate_entropy(values)` | Information content | `-Σ(p × log₂(p))` |
| `calculate_avg_nn_distance(xs, ys)` | Average nearest neighbor | Mean of min distances |
| `calculate_clustering_score(xs, ys)` | Spatial clustering | `1 - (std / mean)` |
| `calculate_iou(bbox1, bbox2)` | Intersection over Union | `intersection / union` |

#### Utility Functions

```python
def calculate_detection_overlaps(detections: List[dict]) -> dict:
    """
    Find all overlapping detection pairs.
    
    Returns:
        {
            'count': int,        # Number of overlapping pairs
            'percentage': float, # % of detections with overlaps
            'avg_iou': float     # Mean IoU of overlaps
        }
    """
```

---

### utils/serialization.py

JSON conversion utilities.

#### `convert_to_json_serializable(obj)`

Recursively converts numpy types to Python native types.

**Conversions:**
| NumPy Type | Python Type |
|------------|-------------|
| `np.integer` | `int` |
| `np.floating` | `float` |
| `np.ndarray` | `list` |
| `np.bool_` | `bool` |

```python
def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types for JSON serialization.
    
    Handles:
    - Dictionaries (recursive)
    - Lists (recursive)
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
│   │   └── Invalid file type
│   └── 500 Internal Server Error
│       ├── Model not loaded
│       ├── Inference failed
│       └── Processing error
└── Standard Exceptions
    ├── FileNotFoundError
    ├── ValueError
    └── RuntimeError
```

### Error Handling Strategy

```python
@app.post("/api/detect")
async def detect_objects(...):
    tmp_path = None
    
    try:
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
```

### Visualization Error Isolation

Each visualization is wrapped individually to prevent cascade failures:

```python
for viz_name, viz_func in visualization_functions.items():
    try:
        visualizations[viz_name] = viz_func()
    except Exception as e:
        print(f"Error generating {viz_name}: {e}")
        visualizations[viz_name] = None
```

### Resource Cleanup

Temporary files are always cleaned up:

```python
finally:
    if tmp_path and os.path.exists(tmp_path):
        os.unlink(tmp_path)
```

---

## ⚡ Performance Optimization

### GPU Memory Management

```python
# At startup - clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Monitor memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
```

### Memory-Efficient Visualizations

```python
# Always close figures after encoding
fig, ax = plt.subplots()
# ... generate chart ...
base64_str = fig_to_base64(fig)
plt.close(fig)  # IMPORTANT: Prevents memory leaks
```

### Response Size Optimization

```python
# Remove large base64 images from saved JSON
json_results = {k: v for k, v in results.items() if k != "visualizations"}

# Save visualizations as separate files
for viz_name, viz_data in visualizations.items():
    save_visualization(viz_data, f"{filename}_{viz_name}.png")
```

### Lazy Model Loading

```python
# Model loaded once at startup, reused for all requests
@app.on_event("startup")
async def startup_event():
    global model
    model = init_detector(config, weights, device)
```

### Performance Benchmarks

| Operation | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| Model loading | 10-15s | 20-30s |
| Single inference | 0.3-0.5s | 2-5s |
| Metrics calculation | 0.1-0.2s | 0.1-0.2s |
| Visualization generation | 1-2s | 1-2s |
| **Total per request** | **1.5-3s** | **4-8s** |

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

### Recommended Security Enhancements

#### API Key Authentication

```python
from fastapi import Security
from fastapi.security.api_key import APIKeyHeader

API_KEY = os.environ.get("RODLA_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(403, "Invalid API key")
    return api_key

@app.post("/api/detect")
async def detect_objects(
    ...,
    api_key: str = Depends(verify_api_key)
):
    ...
```

#### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/detect")
@limiter.limit("10/minute")
async def detect_objects(...):
    ...
```

#### File Size Limits

```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    ...
```

#### Restricted CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
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
├── conftest.py              # Pytest fixtures
├── test_api/
│   ├── test_routes.py       # Endpoint tests
│   └── test_schemas.py      # Pydantic model tests
├── test_services/
│   ├── test_detection.py    # Detection logic tests
│   ├── test_processing.py   # Processing tests
│   └── test_visualization.py # Chart generation tests
├── test_utils/
│   ├── test_helpers.py      # Helper function tests
│   ├── test_metrics.py      # Metrics calculation tests
│   └── test_serialization.py # Serialization tests
└── test_integration/
    └── test_full_pipeline.py # End-to-end tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_utils/test_metrics.py

# Run with verbose output
pytest -v

# Run only fast tests (no model loading)
pytest -m "not slow"
```

### Example Test Cases

```python
# tests/test_utils/test_helpers.py

import pytest
import numpy as np
from utils.helpers import calculate_iou, calculate_skewness

class TestCalculateIoU:
    def test_complete_overlap(self):
        bbox1 = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'width': 100, 'height': 100}
        bbox2 = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'width': 100, 'height': 100}
        assert calculate_iou(bbox1, bbox2) == 1.0
    
    def test_no_overlap(self):
        bbox1 = {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50, 'width': 50, 'height': 50}
        bbox2 = {'x1': 100, 'y1': 100, 'x2': 150, 'y2': 150, 'width': 50, 'height': 50}
        assert calculate_iou(bbox1, bbox2) == 0.0
    
    def test_partial_overlap(self):
        bbox1 = {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'width': 100, 'height': 100}
        bbox2 = {'x1': 50, 'y1': 50, 'x2': 150, 'y2': 150, 'width': 100, 'height': 100}
        iou = calculate_iou(bbox1, bbox2)
        assert 0 < iou < 1

class TestCalculateSkewness:
    def test_symmetric_distribution(self):
        data = [1, 2, 3, 4, 5]
        skew = calculate_skewness(data)
        assert abs(skew) < 0.1  # Nearly symmetric
    
    def test_right_skewed(self):
        data = [1, 1, 1, 1, 10]
        skew = calculate_skewness(data)
        assert skew > 0  # Positive skew
```

### Mocking the Model

```python
# tests/conftest.py

import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_model():
    """Create a mock detection model."""
    model = Mock()
    model.CLASSES = ['paragraph', 'title', 'figure', 'table']
    return model

@pytest.fixture
def mock_detections():
    """Sample detection results."""
    return [
        {
            'class_id': 0,
            'class_name': 'paragraph',
            'bbox': {'x1': 100, 'y1': 100, 'x2': 500, 'y2': 300,
                    'width': 400, 'height': 200, 'center_x': 300, 'center_y': 200},
            'confidence': 0.95,
            'area': 80000,
            'aspect_ratio': 2.0
        }
    ]
```

---

## 🚢 Deployment

### Development Server

```bash
python backend.py
# or
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

### Production with Gunicorn

```bash
gunicorn backend:app -w 1 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --keep-alive 5
```

**Note:** Use `workers=1` for GPU models to avoid memory issues.

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.9 python3-pip

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p outputs

# Expose port
EXPOSE 8000

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
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - RODLA_API_KEY=${RODLA_API_KEY}
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rodla-api
spec:
  replicas: 1
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
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
        volumeMounts:
        - name: outputs
          mountPath: /app/outputs
      volumes:
      - name: outputs
        persistentVolumeClaim:
          claimName: rodla-outputs-pvc
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/rodla-api
upstream rodla_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://rodla_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Failures

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Clear GPU memory before starting
nvidia-smi --gpu-reset

# Or in Python
import torch
torch.cuda.empty_cache()

# Check available GPU memory
nvidia-smi
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
# Check paths in config/settings.py
from pathlib import Path
print(Path(MODEL_CONFIG).exists())  # Should be True
print(Path(MODEL_WEIGHTS).exists())  # Should be True
```

---

#### Inference Errors

**Symptom:** `RuntimeError: Input type and weight type should be the same`

**Solution:**
```python
# Ensure model and input are on same device
model = model.to('cuda')
# or
model = model.to('cpu')
```

**Symptom:** `ValueError: could not broadcast input array`

**Solution:**
```python
# Check image dimensions
from PIL import Image
img = Image.open(image_path)
print(f"Image size: {img.size}")  # Should be reasonable dimensions
```

---

#### Visualization Errors

**Symptom:** `RuntimeError: main thread is not in main loop`

**Solution:**
```python
# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

**Symptom:** Memory usage grows with each request

**Solution:**
```python
# Always close figures after use
fig, ax = plt.subplots()
# ... plotting code ...
plt.savefig(buffer, format='png')
plt.close(fig)  # CRITICAL: Prevents memory leak
plt.close('all')  # Nuclear option if needed
```

---

#### API Errors

**Symptom:** `422 Unprocessable Entity`

**Cause:** Invalid request format

**Solution:**
```bash
# Correct multipart form data format
curl -X POST "http://localhost:8000/api/detect" \
  -H "accept: application/json" \
  -F "file=@image.jpg;type=image/jpeg" \
  -F "score_thr=0.3"
```

**Symptom:** `413 Request Entity Too Large`

**Solution:**
```python
# Increase upload limit in FastAPI
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Or configure in nginx
# client_max_body_size 50M;
```

---

### Debugging Tips

#### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Processing image: {filename}")
logger.debug(f"Detections found: {len(detections)}")
```

#### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

#### Memory Profiling

```python
# Install memory profiler
pip install memory_profiler

# Use decorator
from memory_profiler import profile

@profile
def detect_objects(...):
    ...
```

#### Request Timing

```python
import time

@app.post("/api/detect")
async def detect_objects(...):
    start_time = time.time()
    
    # ... processing ...
    
    elapsed = time.time() - start_time
    logger.info(f"Request completed in {elapsed:.2f}s")
```

---

### Health Checks

```python
# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_used": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" 
            if torch.cuda.is_available() else "N/A"
    }
```

---

## 🤝 Contributing

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

```bash
# Install development dependencies
pip install black isort flake8 mypy

# Format code
black .
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
```

```bash
pip install pre-commit
pre-commit install
```

### Adding New Metrics

1. Create function in appropriate module under `utils/metrics/`
2. Export from `utils/metrics/__init__.py`
3. Call from `services/processing.py`
4. Add to response schema in `api/schemas.py`
5. Document in this README
6. Add tests in `tests/test_utils/test_metrics.py`

### Adding New Visualizations

1. Add function in `services/visualization.py`
2. Call from `generate_comprehensive_visualizations()`
3. Handle errors with try-except
4. Always close figures with `plt.close(fig)`
5. Document chart type in this README

---

## 📚 Citation

If you use this API or the RoDLA model in your research, please cite:

```bibtex
@inproceedings{rodla2024cvpr,
  title={RoDLA: Benchmarking the Robustness of Document Layout Analysis Models},
  author={Author Names},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision 
             and Pattern Recognition (CVPR)},
  year={2024}
}
```

### Related Publications

```bibtex
@article{internimage2023,
  title={InternImage: Exploring Large-Scale Vision Foundation Models 
         with Deformable Convolutions},
  author={Wang et al.},
  journal={CVPR},
  year={2023}
}

@article{dino2022,
  title={DINO: DETR with Improved DeNoising Anchor Boxes 
         for End-to-End Object Detection},
  author={Zhang et al.},
  journal={ICLR},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Support

### Getting Help

- **Documentation:** This README
- **Issues:** [GitHub Issues](https://github.com/yourusername/rodla-api/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/rodla-api/discussions)

### Reporting Bugs

When reporting bugs, please include:

1. Operating system and version
2. Python version
3. GPU model and driver version
4. Complete error traceback
5. Minimal reproducible example
6. Input image (if possible)

### Feature Requests

We welcome feature requests! Please:

1. Check existing issues first
2. Describe the use case
3. Explain expected behavior
4. Provide examples if possible

---

## 🙏 Acknowledgments

- **RoDLA Authors** - For the original model and research
- **MMDetection Team** - For the detection framework
- **InternImage Team** - For the backbone architecture
- **FastAPI** - For the excellent web framework
- **Open Source Community** - For countless contributions

---

<div align="center">

**Built with ❤️ for Document Analysis**

[⬆ Back to Top](#rodla-document-layout-analysis-api)

</div>