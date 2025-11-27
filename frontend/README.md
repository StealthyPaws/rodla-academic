# 🎮 RoDLA 90s Frontend

A retro 90s-themed web interface for the RoDLA Document Layout Analysis system. Single color (teal) design with no gradients, CRT scanlines effect, and authentic terminal-like aesthetics.

## 🎨 Design Features

- **Color Scheme**: Single Teal (#008080) + Lime Green (#00FF00) for authentic 90s terminal feel
- **Theme**: Classic 90s Windows 95/98 inspired interface
- **Effects**: CRT scanlines, blinking text, monospace fonts
- **No Gradients**: Pure, flat 90s design with only one primary color
- **Typography**: MS Sans Serif, Courier New monospace for code
- **Responsive**: Works on mobile, tablet, and desktop

## 📦 Project Structure

```
frontend/
├── index.html          # Main HTML file
├── styles.css          # 90s retro stylesheet
├── script.js           # Frontend JavaScript
├── server.py           # Simple HTTP server
└── README.md          # This file
```

## 🚀 Quick Start

### Option 1: Using Python HTTP Server

```bash
cd frontend
python3 server.py
# Open browser: http://localhost:8080
```

### Option 2: Using Python's Built-in Server

```bash
cd frontend
python3 -m http.server 8080
# Open browser: http://localhost:8080
```

### Option 3: Using Node.js

```bash
cd frontend
npx http-server -p 8080
# Open browser: http://localhost:8080
```

## ⚙️ Prerequisites

### Backend Must Be Running

The frontend expects the RoDLA backend API to be running on `http://localhost:8000`:

```bash
cd deployment/backend
python backend.py
```

Make sure the backend is accessible before using the frontend.

## 🎯 Features

### 1. Document Upload
- Drag and drop interface
- File preview with metadata
- Supported formats: All standard image formats

### 2. Analysis Modes
- **Standard Detection**: Quick object detection
- **Perturbation Analysis**: Test robustness with various perturbations

### 3. Perturbation Types
- Blur
- Noise
- Rotation
- Scaling
- Perspective
- Content Removal

### 4. Real-time Results
- Annotated image with bounding boxes
- Detection statistics
- Class distribution chart
- Detailed detection table
- Performance metrics

### 5. Downloads
- Download annotated image (PNG)
- Download results as JSON

## 🎮 UI Components

### Header
- Application title with 90s style text effects
- System status indicator

### Upload Section
- Drag and drop area
- Image preview with file info

### Analysis Options
- Confidence threshold slider
- Detection mode selector
- Perturbation type selection (when in perturbation mode)

### Results Display
- Annotated image
- Statistics cards (detections, avg confidence, processing time)
- Class distribution bar chart
- Detection details table
- Performance metrics

### Status & Errors
- Real-time status updates with blinking animation
- Error messages with dismiss button

### System Info
- Model information
- Backend status indicator

## 🔧 Configuration

To change the API endpoint, edit `script.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000/api';
```

To modify the color scheme, edit `styles.css`:

```css
:root {
    --primary-color: #008080;      /* Teal */
    --text-color: #00FF00;         /* Lime green */
    --accent-color: #00FFFF;       /* Cyan */
    /* ... */
}
```

## 📱 API Integration

The frontend communicates with the backend via these endpoints:

### Model Info
```
GET /api/model-info
```

### Standard Detection
```
POST /api/detect
- File: image (multipart/form-data)
- score_threshold: float (0-1)
```

### Perturbation Analysis
```
POST /api/detect-with-perturbation
- File: image (multipart/form-data)
- score_threshold: float (0-1)
- perturbation_types: JSON array of strings
```

## 🖥️ Browser Support

- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## ⚡ Performance Tips

1. **Image Size**: Keep images under 10MB for fast processing
2. **Confidence Threshold**: Adjust to reduce false positives
3. **Perturbation Types**: Select only needed perturbation types for faster analysis

## 🐛 Troubleshooting

### Frontend loads but can't connect to backend
- Ensure backend is running: `python backend.py` in deployment/backend
- Check backend is on port 8000
- Check browser console for CORS errors

### Images not displaying
- Check CORS headers are set correctly in the HTTP server
- Verify the image file is valid

### Analysis takes too long
- Reduce image size
- Increase confidence threshold
- Use standard detection instead of perturbation analysis

## 📝 Notes

- All data is processed on the backend, frontend only handles UI
- Results are stored in browser memory during session
- JSON and image downloads are generated client-side

## 🎨 Retro Aesthetic Details

- **CRT Scanlines**: Subtle horizontal lines simulating old monitors
- **Color Usage**: Single teal with lime and cyan accents
- **Borders**: 2px solid borders mimicking Windows 95 style
- **Buttons**: Classic beveled button effect with hover states
- **Font**: Monospace for technical data, sans-serif for UI
- **Animations**: Minimal blinking effects for authentic feel
- **Layout**: Grid-based, box-like sections

## 📞 Support

For issues or questions about the frontend, check the main RoDLA repository.

---

**RoDLA v2.1.0 | 90s Edition | CVPR 2024**
