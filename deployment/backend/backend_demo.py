"""
RoDLA Object Detection API - Demo/Lightweight Backend
Simulates the full backend for testing when real model weights unavailable
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import asyncio

# Initialize FastAPI app
app = FastAPI(
    title="RoDLA Object Detection API (Demo Mode)",
    description="RoDLA Document Layout Analysis API - Demo/Test Version",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model classes
MODEL_CLASSES = [
    'Title', 'Abstract', 'Introduction', 'Related Work', 'Methodology',
    'Experiments', 'Results', 'Discussion', 'Conclusion', 'References',
    'Text', 'Figure', 'Table', 'Header', 'Footer', 'Page Number', 'Caption'
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_demo_detections(image_shape, num_detections=None):
    """Generate realistic demo detections"""
    if num_detections is None:
        num_detections = np.random.randint(8, 20)
    
    height, width = image_shape[:2]
    detections = []
    
    for i in range(num_detections):
        x1 = np.random.randint(10, width - 200)
        y1 = np.random.randint(10, height - 100)
        x2 = x1 + np.random.randint(100, min(300, width - x1))
        y2 = y1 + np.random.randint(50, min(200, height - y1))
        
        detection = {
            'class': np.random.choice(MODEL_CLASSES),
            'confidence': float(np.random.uniform(0.5, 0.99)),
            'box': {
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            }
        }
        detections.append(detection)
    
    return detections

def create_annotated_image(image_array, detections):
    """Create annotated image with bounding boxes"""
    # Convert to PIL Image
    img = Image.fromarray(image_array.astype('uint8'))
    draw = ImageDraw.Draw(img)
    
    # Colors in teal/lime theme
    box_color = (0, 255, 0)  # Lime green
    text_color = (0, 255, 255)  # Cyan
    
    for detection in detections:
        box = detection['box']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        conf = detection['confidence']
        class_name = detection['class']
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        
        # Draw label
        label_text = f"{class_name} {conf*100:.0f}%"
        draw.text((x1, y1-15), label_text, fill=text_color)
    
    return np.array(img)

def apply_perturbation(image_array, perturbation_type):
    """Apply perturbation to image"""
    result = image_array.copy()
    
    if perturbation_type == 'blur':
        result = cv2.GaussianBlur(result, (15, 15), 0)
    
    elif perturbation_type == 'noise':
        noise = np.random.normal(0, 25, result.shape)
        result = np.clip(result.astype(float) + noise, 0, 255).astype(np.uint8)
    
    elif perturbation_type == 'rotation':
        h, w = result.shape[:2]
        center = (w // 2, h // 2)
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h))
    
    elif perturbation_type == 'scaling':
        scale = np.random.uniform(0.8, 1.2)
        h, w = result.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        result = cv2.resize(result, (new_w, new_h))
        # Pad or crop to original size
        if new_h > h or new_w > w:
            result = result[:h, :w]
        else:
            pad_h = h - new_h
            pad_w = w - new_w
            result = cv2.copyMakeBorder(result, pad_h//2, pad_h-pad_h//2, 
                                       pad_w//2, pad_w-pad_w//2, cv2.BORDER_CONSTANT)
    
    elif perturbation_type == 'perspective':
        h, w = result.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [np.random.randint(0, 30), np.random.randint(0, 30)],
            [w - np.random.randint(0, 30), np.random.randint(0, 30)],
            [np.random.randint(0, 30), h - np.random.randint(0, 30)],
            [w - np.random.randint(0, 30), h - np.random.randint(0, 30)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M, (w, h))
    
    return result

def image_to_base64(image_array):
    """Convert image array to base64 string"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# ============================================
# API ENDPOINTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("="*60)
    print("Starting RoDLA Document Layout Analysis API (DEMO)")
    print("="*60)
    print(f"🌐 Main API: http://{API_HOST}:{API_PORT}")
    print(f"📚 Docs: http://localhost:{API_PORT}/docs")
    print(f"📖 ReDoc: http://localhost:{API_PORT}/redoc")
    print("\n🎯 Available Endpoints:")
    print("   • GET  /api/health              - Health check")
    print("   • GET  /api/model-info          - Model information")
    print("   • POST /api/detect              - Standard detection")
    print("   • GET  /api/perturbations/info  - Perturbation info")
    print("   • POST /api/generate-perturbations - Generate perturbations")
    print("   • POST /api/detect-with-perturbation - Detect with perturbations")
    print("="*60)
    print("✅ API Ready! (Demo Mode)\n")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "mode": "demo",
        "timestamp": str(Path.cwd())
    })


@app.get("/api/model-info")
async def model_info():
    """Get model information"""
    return JSONResponse({
        "model_name": "RoDLA InternImage-XL (Demo Mode)",
        "paper": "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models (CVPR 2024)",
        "backbone": "InternImage-XL",
        "detection_framework": "DINO with Channel Attention + Average Pooling",
        "dataset": "M6Doc-P",
        "max_detections_per_image": 300,
        "demo_mode": True,
        "state_of_the_art_performance": {
            "clean_mAP": 70.0,
            "perturbed_avg_mAP": 61.7,
            "mRD_score": 147.6
        }
    })


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), score_threshold: float = Form(0.3)):
    """Standard detection endpoint"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        # Generate demo detections
        detections = generate_demo_detections(image_array.shape)
        
        # Filter by threshold
        detections = [d for d in detections if d['confidence'] >= score_threshold]
        
        # Create annotated image
        annotated = create_annotated_image(image_array, detections)
        annotated_b64 = image_to_base64(annotated)
        
        # Calculate class distribution
        class_dist = {}
        for det in detections:
            cls = det['class']
            class_dist[cls] = class_dist.get(cls, 0) + 1
        
        return JSONResponse({
            "detections": detections,
            "class_distribution": class_dist,
            "annotated_image": annotated_b64,
            "metrics": {
                "total_detections": len(detections),
                "average_confidence": float(np.mean([d['confidence'] for d in detections]) if detections else 0),
                "max_confidence": float(max([d['confidence'] for d in detections]) if detections else 0),
                "min_confidence": float(min([d['confidence'] for d in detections]) if detections else 0)
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/perturbations/info")
async def perturbations_info():
    """Get available perturbation types"""
    return JSONResponse({
        "available_perturbations": [
            "blur",
            "noise", 
            "rotation",
            "scaling",
            "perspective"
        ],
        "description": "Various document perturbations for robustness testing"
    })


@app.post("/api/generate-perturbations")
async def generate_perturbations(
    file: UploadFile = File(...),
    perturbation_types: str = Form("blur,noise")
):
    """Generate and return perturbations"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        # Parse perturbation types
        pert_types = [p.strip() for p in perturbation_types.split(',')]
        
        # Generate perturbations
        results = {
            "original": image_to_base64(image_array),
            "perturbations": {}
        }
        
        for pert_type in pert_types:
            if pert_type:
                perturbed = apply_perturbation(image_array, pert_type)
                results["perturbations"][pert_type] = image_to_base64(perturbed)
        
        return JSONResponse(results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/detect-with-perturbation")
async def detect_with_perturbation(
    file: UploadFile = File(...),
    score_threshold: float = Form(0.3),
    perturbation_types: str = Form("blur,noise")
):
    """Detect with perturbations"""
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        # Parse perturbation types
        pert_types = [p.strip() for p in perturbation_types.split(',')]
        
        # Results for each perturbation
        results = {
            "clean": {},
            "perturbed": {}
        }
        
        # Clean detection
        clean_dets = generate_demo_detections(image_array.shape)
        clean_dets = [d for d in clean_dets if d['confidence'] >= score_threshold]
        clean_img = create_annotated_image(image_array, clean_dets)
        
        results["clean"]["detections"] = clean_dets
        results["clean"]["annotated_image"] = image_to_base64(clean_img)
        
        # Perturbed detections
        for pert_type in pert_types:
            if pert_type:
                perturbed_img = apply_perturbation(image_array, pert_type)
                pert_dets = generate_demo_detections(perturbed_img.shape)
                # Add slight confidence reduction for perturbed
                pert_dets = [
                    {**d, 'confidence': max(0, d['confidence'] - np.random.uniform(0, 0.1))}
                    for d in pert_dets
                ]
                pert_dets = [d for d in pert_dets if d['confidence'] >= score_threshold]
                annotated_pert = create_annotated_image(perturbed_img, pert_dets)
                
                results["perturbed"][pert_type] = {
                    "detections": pert_dets,
                    "annotated_image": image_to_base64(annotated_pert)
                }
        
        return JSONResponse(results)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "="*60)
    print("🛑 Shutting down RoDLA API...")
    print("="*60)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )
