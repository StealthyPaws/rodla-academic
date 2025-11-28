---
title: RoDLA Document Layout Analysis
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# RoDLA: Robust Document Layout Analysis

A document layout analysis tool powered by DINO with InternImage-XL backbone that detects and classifies document elements with robustness testing through 12 types of perturbations.

## Features

✨ **Document Element Detection**
- Detects document sections: text, tables, figures, captions, headers, footers, etc.
- Real-time processing with visual bounding boxes
- Confidence scores for each detection

🔄 **Robustness Testing (12 Perturbation Types)**
- **Blur**: Defocus, Vibration
- **Noise**: Speckle, Texture
- **Content**: Watermark, Image Background
- **Inconsistency**: Ink Holdout, Ink Bleeding, Illumination
- **Spatial**: Rotation, Keystoning, Warping

📊 **Analysis & Metrics**
- Performance statistics (inference time, confidence)
- Class distribution visualization
- Batch processing for multiple images

## Usage

### 1. Upload an Image
- Drag & drop or click to upload a document image (PNG, JPG, WebP)
- Supported formats: JPEG, PNG, WebP
- Max size: 50MB

### 2. Analyze Document
- Click "Analyze" to detect layout elements
- View annotated image with bounding boxes
- See detection confidence scores

### 3. Test Robustness
- Switch to "Perturbation" mode
- Select a perturbation type and degree
- Generate 36 variations (12 types × 3 degrees)
- Analyze perturbed images to test model robustness

## Model Architecture

- **Backbone**: InternImage-XL
- **Detector**: DINO (Deformable Detection Transformer)
- **Framework**: MMDetection
- **Input Size**: 1024x1024
- **Model Weights**: 3.8GB

## Performance

- **GPU**: Optimized for CUDA 11.3
- **CPU**: Graceful fallback with heuristic detection (~73% parity)
- **Inference Time**: ~500-800ms per image (GPU), ~2-3s (CPU)
- **Batch Processing**: Up to 32 images in parallel

## Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: FastAPI, Python 3.8+
- **ML**: PyTorch 1.11.0, MMCV 1.5.0, MMDET 2.28.1
- **Deployment**: Docker, HuggingFace Spaces

## API Endpoints

### Detection
```
POST /api/detect
```
Upload image for document layout analysis.

### Perturbations
```
GET /api/perturbations/info
POST /api/perturb
POST /api/detect-with-perturbation
```
Apply perturbations and test robustness.

### Batch Processing
```
POST /api/detect-batch
GET /api/batch-job/{job_id}
```
Process multiple images.

## Installation (Local)

### Docker
```bash
docker pull xeeshan404/rodla-env:latest
docker run -p 7860:7860 xeeshan404/rodla-env:latest
```

### Manual Setup
```bash
git clone https://github.com/StealthyPaws/rodla-academic.git
cd rodla-academic
pip install -r requirements.txt
python deployment/backend/backend_amar.py
```

## Citation

If you use RoDLA in your research, please cite:

```bibtex
@article{rodla2024,
  title={RoDLA: Robust Document Layout Analysis},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/StealthyPaws/rodla-academic).

---

Built with ❤️ for document understanding
