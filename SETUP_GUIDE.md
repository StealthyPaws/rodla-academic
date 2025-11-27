# 🎮 RoDLA Complete Setup Guide

## 📋 System Overview

This is a Document Layout Analysis system built with:
- **Backend**: FastAPI + PyTorch (RoDLA InternImage-XL model)
- **Frontend**: 90s-themed HTML/CSS/JavaScript interface
- **Design**: Single teal color, no gradients, retro aesthetics

```
┌─────────────────────────────────────────────────────────┐
│          RoDLA Document Layout Analysis                  │
├─────────────────────────────────────────────────────────┤
│  Frontend (90s Theme)  ↔  Backend (FastAPI)             │
│  Port 8080             ↔  Port 8000                     │
│  Browser UI            ↔  Model & Detection             │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Prerequisites

### System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- CUDA 11.3+ (for GPU acceleration)
- Modern web browser

### Required Python Packages
```bash
pip install fastapi uvicorn torch torchvision
```

## 📦 Installation Steps

### Step 1: Clone/Setup Repository

```bash
cd /home/admin/CV/rodla-academic
```

### Step 2: Backend Setup

```bash
cd deployment/backend

# Install dependencies
pip install fastapi uvicorn pillow opencv-python scipy

# Optional: Install GPU support
pip install torch==1.10.2 torchvision==0.11.3 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Step 3: Frontend Setup

```bash
cd frontend

# Frontend requires no installation - it's pure HTML/CSS/JS
# It needs a web server to run
```

## 🚀 Running the System

### Terminal 1: Start the Backend API

```bash
cd deployment/backend
python backend.py
```

Expected output:
```
============================================================
Starting RoDLA Document Layout Analysis API
============================================================
📁 Creating output directories...
   ✓ Main output: outputs
   ✓ Perturbations: outputs/perturbations

🔧 Loading RoDLA model...
...
============================================================
✅ API Ready!
============================================================
🌐 Main API: http://0.0.0.0:8000
📚 Docs: http://localhost:8000/docs
📖 ReDoc: http://localhost:8000/redoc
```

### Terminal 2: Start the Frontend Server

```bash
cd frontend
python3 server.py
```

Expected output:
```
============================================================
🚀 RODLA 90s FRONTEND SERVER
============================================================
📁 Serving from: /home/admin/CV/rodla-academic/frontend
🌐 Server URL: http://localhost:8080
🔗 Open in browser: http://localhost:8080

⚠️  Backend must be running on http://localhost:8000
============================================================
```

### Terminal 3: Open Browser

Open your browser and navigate to:
```
http://localhost:8080
```

## 🎮 Using the Frontend

### 1. Upload Document
- Drag and drop an image into the upload area
- Or click to browse and select
- Supported formats: PNG, JPG, JPEG, GIF, WebP, etc.

### 2. Configure Analysis

**Standard Mode:**
- Adjust confidence threshold (0.0 - 1.0)
- Click [ANALYZE DOCUMENT]

**Perturbation Mode:**
- Select perturbation mode
- Choose which perturbations to apply
- Adjust confidence threshold
- Click [ANALYZE DOCUMENT]

### 3. View Results
- Annotated image with bounding boxes
- Detection count and statistics
- Class distribution chart
- Detailed detection table
- Performance metrics

### 4. Download Results
- Download annotated image as PNG
- Download results as JSON

## 📊 API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Model Info
```bash
curl http://localhost:8000/api/model-info
```

### Standard Detection
```bash
curl -X POST -F "file=@image.jpg" \
     -F "score_threshold=0.3" \
     http://localhost:8000/api/detect
```

### Get Perturbation Info
```bash
curl http://localhost:8000/api/perturbations/info
```

### Detect with Perturbation
```bash
curl -X POST -F "file=@image.jpg" \
     -F "score_threshold=0.3" \
     -F 'perturbation_types=["blur","noise"]' \
     http://localhost:8000/api/detect-with-perturbation
```

## 🎨 Frontend Features

### Visual Design
- **Theme**: 1990s Windows 95/98 inspired
- **Color**: Single teal (#008080) with lime green accents
- **Effects**: CRT scanlines for authentic retro feel
- **Typography**: Monospace fonts for technical data

### Responsive Layout
- Desktop: Full-width optimized
- Tablet: Adjusted for touch
- Mobile: Single column layout

### Key Sections
1. **Header**: Application title and version
2. **Upload Section**: File upload with preview
3. **Options**: Analysis mode and parameters
4. **Status**: Real-time processing status
5. **Results**: Comprehensive analysis results
6. **System Info**: Model and backend information
7. **Footer**: Credits and system status

## 📝 Configuration Files

### Backend Configuration
File: `deployment/backend/config/settings.py`

Key settings:
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_SCORE_THRESHOLD = 0.3
MAX_DETECTIONS_PER_IMAGE = 300
```

### Frontend Configuration
File: `frontend/script.js`

Key settings:
```javascript
const API_BASE_URL = 'http://localhost:8000/api';
```

### Style Configuration
File: `frontend/styles.css`

Key colors:
```css
--primary-color: #008080;      /* Teal */
--text-color: #00FF00;         /* Lime green */
--accent-color: #00FFFF;       /* Cyan */
--bg-color: #000000;           /* Black */
```

## 🐛 Troubleshooting

### Issue: Frontend can't connect to backend
**Solution:**
1. Verify backend is running: `http://localhost:8000`
2. Check for CORS errors in browser console
3. Ensure both are on the same machine or network

### Issue: Backend fails to load model
**Solution:**
1. Check model weights file exists
2. Verify PyTorch/CUDA installation
3. Check Python path configuration

### Issue: Analysis takes very long
**Solution:**
1. Use GPU acceleration if available
2. Reduce image resolution
3. Increase confidence threshold

### Issue: Port already in use
**Solution:**
```bash
# Change frontend port
python3 -m http.server 8081

# Or kill existing process
lsof -ti :8080 | xargs kill -9
```

## 📚 Project Structure

```
rodla-academic/
├── deployment/
│   └── backend/
│       ├── backend.py           # Main API server
│       ├── config/
│       │   └── settings.py      # Configuration
│       ├── api/
│       │   ├── routes.py        # API endpoints
│       │   └── schemas.py       # Data models
│       ├── services/            # Business logic
│       ├── core/                # Core functionality
│       ├── perturbations/       # Perturbation methods
│       ├── utils/               # Utilities
│       └── tests/               # Test suite
│
├── frontend/
│   ├── index.html               # Main page
│   ├── styles.css               # 90s stylesheet
│   ├── script.js                # Frontend logic
│   ├── server.py                # HTTP server
│   └── README.md                # Frontend docs
│
└── model/                        # Model configurations
    └── configs/                 # Detection configs
```

## 🔄 Workflow Example

1. **Start Backend**: `python backend.py`
2. **Start Frontend**: `python3 server.py`
3. **Open Browser**: Navigate to `http://localhost:8080`
4. **Upload Image**: Drag and drop or click to select
5. **Analyze**: Click [ANALYZE DOCUMENT]
6. **View Results**: See detections and metrics
7. **Download**: Export image or JSON results

## 📈 Performance Metrics

- **Detection Speed**: ~3-5 seconds per image (GPU)
- **Detection Accuracy**: mAP 70.0 (clean), 61.7 (average perturbed)
- **Max Image Size**: 50MB
- **Max Detections**: 300 per image
- **Batch Processing**: Up to 300 images per batch

## 🔐 Security Notes

- Frontend: Client-side processing only, no data stored
- Backend: File uploads limited to 50MB
- CORS: Enabled for development (modify in production)
- No authentication: Use firewall/proxy in production

## 🎓 Model Information

- **Model Name**: RoDLA InternImage-XL
- **Paper**: CVPR 2024
- **Backbone**: InternImage-XL
- **Detection Framework**: DINO with Channel Attention
- **Training Dataset**: M6Doc-P
- **Robustness Focus**: Perturbation resilience

## 📞 Getting Help

1. Check backend logs for detailed error messages
2. Check browser console for frontend errors
3. Review API documentation at `http://localhost:8000/docs`
4. Check GitHub issues for known problems

## 🎉 Success Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 8080
- [ ] Browser can load `http://localhost:8080`
- [ ] Can upload test image
- [ ] Analysis completes successfully
- [ ] Results display correctly

## 📅 Next Steps

1. **Test with Sample Images**: Try various document types
2. **Adjust Thresholds**: Optimize for your use case
3. **Explore Perturbations**: Test robustness features
4. **Deploy**: Follow deployment guide for production use
5. **Integrate**: Connect with your applications

---

**RoDLA v2.1.0 | 90s Edition | CVPR 2024**

For more information, visit the main README.md and project homepage.
