# 🎮 RoDLA 90s Edition - Quick Test Guide

## ✅ System Status

**Backend**: Running on http://localhost:8000 ✓
**Frontend**: Running on http://localhost:8080 ✓

## 🚀 Access the Frontend

Open your browser and navigate to:
```
http://localhost:8080
```

## 🎯 Step-by-Step Usage Guide

### 1. Upload a Document Image

**What to do:**
- Drag and drop an image file onto the upload area
- Or click the upload box to browse and select a file

**Supported formats:**
- PNG, JPG, JPEG, GIF, WebP, TIFF, BMP

**Example:** You can use any document image like:
- Screenshots of documents
- Scanned pages
- Photos of papers
- Digital document images

### 2. Generate Perturbations (Preview Features)

**Step 1: Switch to Perturbation Mode**
- After uploading an image, look at the "Detection Mode" buttons
- Click the `PERTURBATION` button

**Step 2: Select Perturbation Types**
- You'll see checkboxes for different perturbations:
  - ✓ BLUR (default checked)
  - ✓ NOISE (default checked)
  - ✓ ROTATION (default checked)
  - ✓ SCALING (default checked)
  - ✓ PERSPECTIVE (default checked)

**Step 3: Click "GENERATE PERTURBATIONS"**
- A new button will appear: `[GENERATE PERTURBATIONS]`
- Click it to see the original and perturbed versions side by side

**Result:**
- You'll see:
  - Original image (untouched)
  - Blur version
  - Noise version
  - Rotation version
  - Scaling version
  - Perspective version

### 3. Analyze with Standard Detection

**Step 1: Switch Back to Standard Mode**
- Click the `STANDARD` button under Detection Mode

**Step 2: Adjust Confidence Threshold (Optional)**
- Use the slider to adjust detection confidence
- Range: 0.0 to 1.0
- Default: 0.3
- Higher = fewer but more confident detections

**Step 3: Click "ANALYZE DOCUMENT"**
- The system will run detection on your image

**Results You'll See:**
- **Annotated Image**: Your image with bounding boxes showing detected elements
- **Detection Count**: Total number of elements found
- **Average Confidence**: Average confidence score
- **Processing Time**: How long it took
- **Class Distribution**: Bar chart showing types of elements detected
- **Detection Table**: Detailed list of each detection
- **Metrics**: Performance statistics

### 4. Analyze with Perturbations

**Step 1: Switch to Perturbation Mode**
- Click the `PERTURBATION` button

**Step 2: Select Perturbation Types**
- Choose which perturbations to test
- You can uncheck types you don't want

**Step 3: Click "ANALYZE DOCUMENT"**
- System will:
  1. Detect on the clean (original) image
  2. Apply each selected perturbation
  3. Detect on each perturbed version
  4. Compare results

**Results You'll See:**
- Clean detection results
- Detection results for each perturbation type
- Comparison showing robustness

### 5. Download Results

**Download Annotated Image:**
- Click `[DOWNLOAD IMAGE]`
- Saves as PNG with bounding boxes

**Download JSON Results:**
- Click `[DOWNLOAD JSON]`
- Contains all detection details and metadata

### 6. Clear and Start Over

**Click "CLEAR ALL"** to:
- Reset the form
- Clear the preview
- Remove all results
- Start fresh with a new image

## 🎨 Interface Elements Explained

### Status Indicators
- `● ONLINE` = Backend is connected and ready
- `● DEMO MODE` = Using demo data (backend not available)
- Blinking text = Processing in progress

### Color Scheme (90s Theme)
- **Lime Green** (#00FF00): Text and highlights
- **Teal** (#008080): Borders and primary elements
- **Cyan** (#00FFFF): Accents and labels
- **Black** (#000000): Background (CRT effect)
- **Red** (#FF0000): Error messages

### Sections

1. **Header** - Application title and version
2. **Upload Section** - File upload and preview
3. **Analysis Options** - Mode selection and settings
4. **Perturbations Preview** - Visual comparison of original and perturbed images
5. **Status** - Real-time processing updates
6. **Results** - Detection visualizations and data
7. **System Info** - Model and backend information

## 🔧 Troubleshooting

### Issue: Can't upload file
**Solution:**
- Ensure file is an image (PNG, JPG, etc.)
- Check file size (max 50MB)
- Try drag & drop instead of clicking

### Issue: Analysis is slow
**Solution:**
- Image is too large - try a smaller image
- System is processing - wait for completion
- Close other applications to free resources

### Issue: No results displayed
**Solution:**
- Check browser console (F12) for errors
- Ensure backend is running (check port 8000)
- Try refreshing the page

### Issue: Backend not responding
**Solution:**
```bash
# Check if backend is running:
curl http://localhost:8000/api/health

# If not running, restart it:
cd /home/admin/CV/rodla-academic/deployment/backend
python3 backend_demo.py
```

## 📊 What the System Does

### Detection
The system identifies and classifies layout elements in documents:
- **Text blocks**: Paragraphs and text content
- **Tables**: Structured data
- **Figures**: Images and diagrams
- **Headers/Footers**: Document meta-elements
- **Titles**: Main document titles
- And more document layout elements

### Perturbation
Tests document robustness by applying realistic distortions:
- **Blur**: Simulates camera focus issues
- **Noise**: Adds graininess (compression, scanning artifacts)
- **Rotation**: Skewed/rotated documents
- **Scaling**: Different document scales
- **Perspective**: Angled camera views

### Robustness Analysis
Measures how well the system handles:
- Clean documents (baseline)
- Degraded/perturbed documents
- Real-world variations

## 🎓 Model Information

**Model**: RoDLA InternImage-XL
- **Paper**: CVPR 2024
- **Framework**: PyTorch + FastAPI
- **Backbone**: InternImage-XL neural network
- **Detection Type**: DINO (Deformable Instance-aware Object detection)
- **Performance**:
  - Clean documents: mAP 70.0
  - Perturbed documents: mAP 61.7
  - Robustness Score: 147.6

## 💡 Tips for Best Results

1. **Use Quality Images**
   - Clear, well-lit scans or photos
   - Good contrast
   - Reasonable resolution (not too small, not huge)

2. **Adjust Threshold**
   - Lower (0.1-0.3): More detections, more false positives
   - Higher (0.7-0.9): Fewer detections, higher confidence

3. **Compare Modes**
   - Run standard detection first to establish baseline
   - Then run perturbation to see robustness
   - Compare results side-by-side

4. **Try Different Perturbations**
   - Start with blur and noise
   - Then try rotation and perspective
   - Combine multiple perturbations

## 📈 Performance Notes

- **Detection Speed**: 2-5 seconds per image
- **Perturbation Generation**: 1-2 seconds per perturbation type
- **Max Batch Size**: 300 images (via batch API)
- **Max File Size**: 50 MB
- **Max Detections**: 300 per image

## 🔗 API Endpoints (Advanced)

For developers integrating with the backend:

```bash
# Health check
curl http://localhost:8000/api/health

# Model info
curl http://localhost:8000/api/model-info

# Standard detection
curl -X POST -F "file=@image.jpg" \
     -F "score_threshold=0.3" \
     http://localhost:8000/api/detect

# Generate perturbations
curl -X POST -F "file=@image.jpg" \
     -F "perturbation_types=blur,noise" \
     http://localhost:8000/api/generate-perturbations

# Detect with perturbations
curl -X POST -F "file=@image.jpg" \
     -F "score_threshold=0.3" \
     -F "perturbation_types=blur,noise" \
     http://localhost:8000/api/detect-with-perturbation

# API Docs
open http://localhost:8000/docs
```

## 🎮 Demo Mode Features

The system includes automatic demo mode that:
- Works without backend (uses simulated data)
- Shows all UI features and workflows
- Perfect for testing and demonstrations
- Status shows `● DEMO MODE` when active

## ✨ 90s UI Features

- **CRT Scanlines**: Retro monitor effect
- **Terminal Style**: Green text on black background
- **Classic Borders**: Windows 95/98 style
- **Monospace Fonts**: Authentic retro look
- **No Gradients**: Pure flat 90s design
- **Blinking Elements**: Status indicators
- **Keyboard-Friendly**: Tab navigation works

## 📞 Getting Help

1. Check this guide first
2. Review browser console (F12 > Console tab)
3. Check backend logs: `tail -f /tmp/backend.log`
4. Check frontend server output
5. Review the main README.md

---

## 🎉 Quick Start Summary

```bash
# 1. Open browser
firefox http://localhost:8080

# 2. Upload an image
# (Drag & drop or click to select)

# 3. Generate perturbations (optional)
# - Select "PERTURBATION" mode
# - Check perturbation types
# - Click "[GENERATE PERTURBATIONS]"
# - View original vs perturbed images

# 4. Run analysis
# - Click "[ANALYZE DOCUMENT]"
# - Wait for processing
# - View results with bounding boxes

# 5. Download results
# - Click "[DOWNLOAD IMAGE]" or "[DOWNLOAD JSON]"
# - Save to your computer
```

**That's it! Enjoy the 90s retro interface with modern document analysis! 🎮**

---

RoDLA v2.1.0 | 90s Edition | CVPR 2024
