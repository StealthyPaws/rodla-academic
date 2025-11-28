# 🎮 RoDLA 90s Frontend - Complete Project Documentation

## 📊 Project Analysis Summary

### What is RoDLA?

**RoDLA** (Robust Document Layout Analysis) is a state-of-the-art computer vision system for detecting and classifying layout elements in document images. It was published at **CVPR 2024** and focuses on robustness testing with various perturbations.

**Key Features:**
- Document element detection (text, tables, figures, headers, footers, etc.)
- Robustness testing with perturbations (blur, noise, rotation, scaling, perspective)
- mAP Score: 70.0 on clean documents, 61.7 on average perturbed
- mRD (Robustness Degradation) Score: 147.6
- Model: InternImage-XL backbone with DINO detection framework

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RoDLA System (90s Edition)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐              ┌──────────────────┐   │
│  │   Frontend       │  (HTTP)      │   Backend        │   │
│  │  90s Terminal    │──────────────│   FastAPI        │   │
│  │  Port: 8080      │  (JSON/Image)│   Port: 8000     │   │
│  └──────────────────┘              └──────────────────┘   │
│         │                                    │             │
│         │                                    ▼             │
│         │                          ┌──────────────────┐   │
│         │                          │   PyTorch Model  │   │
│         │                          │   InternImage-XL │   │
│         │                          └──────────────────┘   │
│         │                                    │             │
│         └────────────────────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 Frontend Design

### Color Scheme
- **Primary Color**: Teal (#008080)
- **Text Color**: Lime Green (#00FF00)
- **Accent Color**: Cyan (#00FFFF)
- **Background**: Black (#000000)
- **Error Color**: Red (#FF0000)
- **No Gradients**: Pure flat 90s design

### Design Elements
✓ CRT Scanlines effect
✓ Blinking status animations
✓ Classic Windows 95/98 style borders
✓ Monospace fonts (Courier New for data)
✓ MS Sans Serif for UI
✓ Terminal-like interface

### Responsive Breakpoints
- Desktop: Full-width optimized
- Tablet (768px): Adjusted grid layouts
- Mobile (< 768px): Single column, touch-friendly

## 📁 Project Structure

```
rodla-academic/
│
├── SETUP_GUIDE.md              # Complete setup documentation
├── PROJECT_ANALYSIS.md         # This file
├── start.sh                    # Startup script (both services)
│
├── frontend/                   # 90s-themed Web UI
│   ├── index.html             # Main page
│   ├── styles.css             # Retro stylesheet (1000+ lines)
│   ├── script.js              # Frontend logic + demo mode
│   ├── server.py              # Python HTTP server
│   └── README.md              # Frontend documentation
│
├── deployment/
│   └── backend/               # FastAPI backend
│       ├── backend.py         # Main server
│       ├── config/
│       │   └── settings.py    # Configuration
│       ├── api/
│       │   ├── routes.py      # API endpoints
│       │   └── schemas.py     # Data models
│       ├── core/              # Core functionality
│       ├── services/          # Business logic
│       ├── perturbations/     # Perturbation methods
│       ├── utils/             # Utilities
│       └── tests/             # Test suite
│
├── model/                      # ML Model
│   ├── configs/               # Model configs
│   ├── ops_dcnv3/             # CUDA operations
│   └── train.py / test.py    # Training/testing
│
└── perturbation/              # Perturbation tools
    └── *.py                   # Various perturbation methods
```

## Datasets

### Training
Download the RoDLA dataset from Google Driver to the desired root directory for training.
  - [PubLayNet-P](https://drive.google.com/file/d/1bfjaxb5fAjU7sFqtM3GfNYm0ynrB5Vwo/view?usp=drive_link)

### Finetuning

Download the RoDLA dataset from Google Driver to the desired root directory for finetuning.
  - [DockBank](https://drive.google.com/drive/folders/1h0lda3t2vXO-jp8-XgHtLcyXcMX1LT-9?usp=sharing)


## Weights

### Training
Download the weights of model pretrained on PubLayNet-P from Google Driver.
 - [Checkpoints for PubLayNet](https://drive.google.com/file/d/1I2CafA-wRKAWCqFgXPgtoVx3OQcRWEjp/view?usp=sharing)

### Finetuning

Download the weights of model finetuned on DocBank from Google Driver.
  - [Checkpoints for DocBank](https://drive.google.com/file/d/1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz/view?usp=drive_link)



## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)

```bash
cd /home/admin/CV/rodla-academic
./start.sh
```

This script will:
1. Check system requirements
2. Start backend API on port 8000
3. Start frontend server on port 8080
4. Display access points and logs

### Option 2: Manual Startup

**Terminal 1 - Backend:**
```bash
cd /home/admin/CV/rodla-academic/deployment/backend
python backend.py
```

**Terminal 2 - Frontend:**
```bash
cd /home/admin/CV/rodla-academic/frontend
python3 server.py
```

**Terminal 3 - Browser:**
```
Open: http://localhost:8080
```

### Option 3: Alternative HTTP Servers

```bash
cd /home/admin/CV/rodla-academic/frontend

# Using http.server
python3 -m http.server 8080

# Using npx http-server
npx http-server -p 8080 -c-1

# Using PHP
php -S localhost:8080
```

## 🎮 User Interface Guide

### Main Sections

#### 1. Header
```
┌──────────────────────────────────────┐
│          RoDLA                       │
│  >>> DOCUMENT LAYOUT ANALYSIS <<<   │
│     [VERSION 2.1.0 - 90s EDITION]   │
└──────────────────────────────────────┘
```
- Application branding
- Version information
- Status indicator

#### 2. Upload Section
- Drag & Drop Area
- File preview with metadata
- Supported: All standard image formats

#### 3. Analysis Options
- **Confidence Threshold**: 0.0 - 1.0 slider
- **Detection Mode**: Standard or Perturbation
- **Perturbation Types** (if perturbation mode selected):
  - Blur
  - Noise
  - Rotation
  - Scaling
  - Perspective
  - Content Removal

#### 4. Action Buttons
- `[ANALYZE DOCUMENT]` - Run analysis
- `[CLEAR ALL]` - Reset form

#### 5. Status Display
- Real-time status updates
- Progress bar
- Blinking animation

#### 6. Results Display
When analysis completes:
- **Annotated Image**: Detection visualization
- **Statistics Cards**: Count, confidence, time
- **Class Distribution**: Bar chart
- **Detection Table**: Detailed detection list
- **Metrics Box**: Performance metrics
- **Download Options**: Image & JSON exports

#### 7. System Info
- Model information
- Backend status
- Online/Demo mode indicator

### Workflow Example

```
1. Upload Image
   └─ Preview shown
      └─ Analyze button enabled

2. Configure Options
   └─ Set threshold
   └─ Choose mode
   └─ Select perturbations (if needed)

3. Click Analyze
   └─ Status shows progress
   └─ Backend processes image
   └─ Results displayed

4. Review Results
   └─ View annotated image
   └─ Check statistics
   └─ Review detections table

5. Download
   └─ Save annotated image (PNG)
   └─ Save detailed results (JSON)

6. Reset for Next Image
   └─ Click Clear All
   └─ Upload new image
```

## 🔌 API Integration

### Backend Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/model-info` | Model information |
| POST | `/api/detect` | Standard detection |
| GET | `/api/perturbations/info` | Perturbation info |
| POST | `/api/detect-with-perturbation` | Detection with perturbations |
| POST | `/api/batch` | Batch processing |

### Request/Response Format

#### Standard Detection
**Request:**
```json
{
  "file": "image_file",
  "score_threshold": 0.3
}
```

**Response:**
```json
{
  "detections": [
    {
      "class": "Text",
      "confidence": 0.95,
      "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}
    }
  ],
  "class_distribution": {"Text": 5, "Table": 2},
  "annotated_image": "base64_encoded_image",
  "metrics": {}
}
```

## 💡 Features

### Standard Detection
- Real-time object detection
- Bounding box generation
- Confidence scoring
- Class classification

### Perturbation Analysis
- Apply 1+ perturbation types
- Test robustness
- Benchmark degradation
- Compare clean vs. perturbed

### Visualization
- Annotated images with boxes
- Color-coded labels
- Confidence indicators
- Class distributions

### Download Options
- PNG images (with annotations)
- JSON data (full results)
- Timestamp metadata

## 🎯 Demo Mode

If the backend is unavailable, the frontend automatically switches to **Demo Mode**:

✓ Works without backend running
✓ Generates realistic sample data
✓ Shows 90s UI functionality
✓ Perfect for demonstration
✓ No network required

**Status Indicator Changes to: `● DEMO MODE` (Yellow)**

## ⚙️ Configuration

### Backend Configuration

File: `deployment/backend/config/settings.py`

```python
API_HOST = "0.0.0.0"           # Listen on all interfaces
API_PORT = 8000                 # API port
DEFAULT_SCORE_THRESHOLD = 0.3   # Default confidence threshold
MAX_DETECTIONS_PER_IMAGE = 300  # Max results per image
```

### Frontend Configuration

File: `frontend/script.js`

```javascript
const API_BASE_URL = 'http://localhost:8000/api';  // Backend URL
```

### Style Configuration

File: `frontend/styles.css`

```css
:root {
    --primary-color: #008080;      /* Teal */
    --text-color: #00FF00;         /* Lime */
    --accent-color: #00FFFF;       /* Cyan */
    --bg-color: #000000;           /* Black */
}
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Detection Speed (GPU) | 3-5 seconds/image |
| Detection Speed (CPU) | 10-15 seconds/image |
| Model mAP (Clean) | 70.0 |
| Model mAP (Perturbed Avg) | 61.7 |
| mRD Score | 147.6 |
| Max Batch Size | 300 images |
| Max File Size | 50 MB |
| Max Detections | 300 per image |

## 🐛 Troubleshooting

### Frontend loads but can't connect
```
✗ Backend not running
  → Start: cd deployment/backend && python backend.py

✗ Wrong port
  → Check config: API_BASE_URL in script.js

✗ CORS error
  → Backend CORS misconfigured
  → Check settings.py CORS_ORIGINS
```

### Analysis takes too long
```
✗ Image too large
  → Reduce image size/resolution

✗ CPU processing (no GPU)
  → Install PyTorch with CUDA
  → Or increase patience

✗ Multiple analyses queued
  → Wait for current to finish
```

### Port already in use
```bash
# Find what's using port 8000/8080
lsof -ti :8000 | xargs kill -9
lsof -ti :8080 | xargs kill -9

# Or use different port
python3 -m http.server 8081
```

## 🔒 Security Considerations

### Frontend
- No sensitive data stored locally
- All processing on backend
- Client-side download only

### Backend
- File upload limits (50MB)
- No direct file system access
- Input validation
- CORS restrictions (configure for production)

### Deployment
- Use HTTPS in production
- Implement authentication
- Rate limiting
- File type validation

## 📝 Browser Support

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 90+ | ✓ Fully supported |
| Firefox | 88+ | ✓ Fully supported |
| Safari | 14+ | ✓ Fully supported |
| Edge | 90+ | ✓ Fully supported |
| IE 11 | - | ✗ Not supported |

## 🎓 Model Details

### Architecture
- **Backbone**: InternImage-XL
- **Detection Framework**: DINO (Deformable INstance-aware Object detection)
- **Attention**: Channel Attention + Average Pooling
- **Pre-training**: ImageNet-22K

### Training Data
- **Primary**: PubLayNet (perturbed PubLayNet-P dataset)
- **Test**: PubLayNet-P, DocLayNet-P (perturbed variants)
- **Augmentation**: 450,000+ perturbed documents

### Detection Classes
Varies by model, typically includes:
- Text blocks
- Tables
- Figures
- Headers
- Footers
- Page numbers
- Captions

## 🚀 Deployment Options

### Local Development
```bash
./start.sh
```

### Docker Deployment
```dockerfile
# Dockerfile (example)
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000 8080
CMD ["./start.sh"]
```

### Production Deployment
1. Use HTTPS/SSL
2. Implement authentication
3. Add rate limiting
4. Use production WSGI server
5. Configure CORS properly
6. Add monitoring/logging

## 📚 References

- **Paper**: RoDLA: Benchmarking the Robustness of Document Layout Analysis Models (CVPR 2024)
- **Framework**: FastAPI, PyTorch, OpenCV
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **License**: Apache 2.0

## 🎉 Success Indicators

When everything is working correctly:

✓ Backend starts without errors
✓ Frontend loads at http://localhost:8080
✓ Can upload image files
✓ Analysis completes and displays results
✓ Can download results as PNG and JSON
✓ Results include annotations with bounding boxes
✓ Status shows "● ONLINE" (or "● DEMO MODE" for demo)

## 📞 Getting Help

1. **Check Documentation**: Read README files
2. **Review Logs**: Check /tmp/rodla_*.log files
3. **Browser Console**: Open DevTools (F12) for errors
4. **API Docs**: Visit http://localhost:8000/docs
5. **GitHub Issues**: Check project repository

## 🎨 Future Enhancements

Potential additions:
- [ ] Multiple model selection
- [ ] Batch processing UI
- [ ] Real-time preview
- [ ] Advanced filtering
- [ ] Export to COCO format
- [ ] Database integration
- [ ] WebSocket support
- [ ] Progressive image uploads

---

## 🎯 Summary

**RoDLA 90s Edition** provides:

✅ **Retro 90s Interface**: Single color, no gradients, authentic styling
✅ **Complete Backend**: FastAPI with PyTorch model
✅ **Demo Mode**: Works without backend connection
✅ **Responsive Design**: Mobile, tablet, desktop support
✅ **Production Ready**: Error handling, logging, configuration
✅ **Easy to Use**: Simple drag-and-drop interface
✅ **Comprehensive Results**: Visualizations and metrics
✅ **Download Support**: PNG images and JSON data

**RoDLA v2.1.0 | 90s Edition | CVPR 2024**

Created with ❤️ for retro computing enthusiasts and document analysis professionals.
