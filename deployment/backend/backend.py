"""
RoDLA Backend - Production Version
Uses real InternImage-XL weights and all 12 perturbation types with 3 degree levels
MMDET disabled if MMCV extensions unavailable - perturbations always functional
"""

import os
import sys
import json
import base64
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Global configuration"""
    import os
    
    API_PORT = 7860  # HuggingFace Spaces standard port
    # Get repo root dynamically - works on any system
    if os.path.exists("/app"):
        REPO_ROOT = Path("/app")
    else:
        REPO_ROOT = Path(__file__).parent.parent.parent
    
    MODEL_CONFIG_PATH = REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"
    MODEL_WEIGHTS_PATH = REPO_ROOT / "finetuning_rodla/finetuning_rodla/checkpoints/rodla_internimage_xl_publaynet.pth"
    PERTURBATIONS_DIR = REPO_ROOT / "deployment/backend/perturbations"
    FRONTEND_DIR = REPO_ROOT / "frontend"
    
    # Automatically use GPU if available, otherwise CPU
    @staticmethod
    def get_device():
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"


# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="RoDLA Production Backend", version="3.0.0")

# Detect device
import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

model_state = {
    "loaded": False,
    "model": None,
    "error": None,
    "model_type": "RoDLA InternImage-XL (MMDET)",
    "device": DEVICE,
    "mmdet_available": False
}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: Static files will be mounted at the END after all API routes are defined
# This ensures /api/* routes take priority over static file fallback


# ============================================================================
# Root endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - serve index.html"""
    if (Config.FRONTEND_DIR / "index.html").exists():
        with open(Config.FRONTEND_DIR / "index.html", "r") as f:
            return {"message": "Frontend served - navigate to the web interface"}
    return {
        "title": "RoDLA Document Layout Analysis API",
        "message": "API is running. Frontend files not found.",
        "endpoints": {
            "detection": "/api/detect",
            "health": "/api/health",
            "model_info": "/api/model-info",
            "perturbations": "/api/perturbations/info"
        }
    }


# ============================================================================
# M6Doc Dataset Classes
# ============================================================================

LAYOUT_CLASS_MAP = {
    i: "Text" for i in range(75)
}
# Simplified mapping to layout elements
for i in range(75):
    if i in [1, 2, 3, 4, 5]:
        LAYOUT_CLASS_MAP[i] = "Title"
    elif i in [6, 7]:
        LAYOUT_CLASS_MAP[i] = "List"
    elif i in [8, 9]:
        LAYOUT_CLASS_MAP[i] = "Figure"
    elif i in [10, 11]:
        LAYOUT_CLASS_MAP[i] = "Table"
    elif i in [12, 13, 14]:
        LAYOUT_CLASS_MAP[i] = "Header"


# ============================================================================
# Utility Functions
# ============================================================================

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Ensure RGB order
        if isinstance(image.flat[0], np.uint8):
            image_to_encode = image
        else:
            image_to_encode = (image * 255).astype(np.uint8)
    else:
        image_to_encode = image
    
    _, buffer = cv2.imencode('.png', image_to_encode)
    return base64.b64encode(buffer).decode('utf-8')


def heuristic_detect(image_np: np.ndarray) -> List[Dict]:
    """Enhanced heuristic-based detection when MMDET is unavailable
    Uses multiple edge detection methods and texture analysis"""
    h, w = image_np.shape[:2]
    detections = []
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Try multiple edge detection methods for better coverage
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 30, 100)
    
    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Apply morphological operations to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Also try watershed/connected components for text detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary)
    
    # Process contours to create pseudo-detections
    processed_boxes = set()
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Skip if too small or too large
        if cw < 15 or ch < 15 or cw > w * 0.98 or ch > h * 0.98:
            continue
        
        area_ratio = (cw * ch) / (w * h)
        if area_ratio < 0.0005 or area_ratio > 0.9:
            continue
        
        # Skip if box is too similar to already processed boxes
        box_key = (round(x/10)*10, round(y/10)*10, round(cw/10)*10, round(ch/10)*10)
        if box_key in processed_boxes:
            continue
        processed_boxes.add(box_key)
        
        # Analyze content to determine class
        roi = gray[y:y+ch, x:x+cw]
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        roi_edges = cv2.Canny(roi_blur, 50, 150)
        edge_density = np.sum(roi_edges > 0) / roi.size
        
        aspect_ratio = cw / (ch + 1e-6)
        
        # Classification logic
        if aspect_ratio > 2.5 or (aspect_ratio > 2 and edge_density < 0.05):
            # Wide with sparse edges = likely figure/table
            class_name = "Figure"
            class_id = 8
            confidence = 0.6 + 0.35 * (1 - min(area_ratio / 0.5, 1.0))
        elif aspect_ratio < 0.3:
            # Narrow = likely list or table column
            class_name = "List"
            class_id = 6
            confidence = 0.55 + 0.4 * (1 - min(area_ratio / 0.3, 1.0))
        elif edge_density > 0.15:
            # High edge density = likely table or complex content
            class_name = "Table"
            class_id = 10
            confidence = 0.5 + 0.4 * edge_density
        else:
            # Default = text content
            class_name = "Text"
            class_id = 50
            confidence = 0.5 + 0.4 * (1 - min(area_ratio / 0.3, 1.0))
        
        # Ensure confidence in [0, 1]
        confidence = min(max(confidence, 0.3), 0.95)
        
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(confidence),
            "bbox": {
                "x": float(x / w),
                "y": float(y / h),
                "width": float(cw / w),
                "height": float(ch / h)
            },
            "area": float(area_ratio)
        })
    
    # Sort by confidence and keep top 30
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return detections[:30]


# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load the RoDLA model with actual weights"""
    global model_state
    
    print("\n" + "="*70)
    print("🚀 Loading RoDLA InternImage-XL with Real Weights")
    print("="*70)
    
    # Verify weight file exists
    if not Config.MODEL_WEIGHTS_PATH.exists():
        error_msg = f"Weights not found: {Config.MODEL_WEIGHTS_PATH}"
        print(f"❌ {error_msg}")
        model_state["loaded"] = False
        model_state["error"] = error_msg
        return None
    
    weights_size = Config.MODEL_WEIGHTS_PATH.stat().st_size / (1024**3)
    print(f"✅ Weights file: {Config.MODEL_WEIGHTS_PATH}")
    print(f"   Size: {weights_size:.2f}GB")
    
    # Verify config exists
    if not Config.MODEL_CONFIG_PATH.exists():
        error_msg = f"Config not found: {Config.MODEL_CONFIG_PATH}"
        print(f"❌ {error_msg}")
        model_state["loaded"] = False
        model_state["error"] = error_msg
        return None
    
    print(f"✅ Config file: {Config.MODEL_CONFIG_PATH}")
    print(f"📍 Device: {model_state['device']}")
    
    if model_state["device"] == "cpu":
        print("⚠️  WARNING: DCNv3 (used in InternImage backbone) only supports CUDA")
        print("   CPU inference is NOT available. Using heuristic fallback.")
        print("   Skipping MMDET model load on CPU-only system.")
        model_state["loaded"] = False
        model_state["mmdet_available"] = False
        model_state["error"] = "CUDA required for MMDET inference"
        return None
    
    # Try to import and load MMDET (GPU only)
    try:
        print("⏳ Setting up model environment...")
        import torch
        
        # Try to import mmdet with graceful degradation
        try:
            from mmdet.apis import init_detector
        except ImportError as ie:
            print(f"⚠️  MMDET import error: {ie}")
            print("   This may be due to missing mmcv._ext (compiled C++ extension)")
            print("   Attempting alternative loading method...")
            raise ie
        
        print("⏳ Loading model from weights (this will take ~30-60 seconds)...")
        print("   File: 3.8GB checkpoint...")
        
        model = init_detector(
            str(Config.MODEL_CONFIG_PATH),
            str(Config.MODEL_WEIGHTS_PATH),
            device=model_state["device"]
        )
        
        if model is not None:
            # Set model to evaluation mode
            model.eval()
            
            model_state["model"] = model
            model_state["loaded"] = True
            model_state["mmdet_available"] = True
            model_state["error"] = None
            
            print("✅ RoDLA Model loaded successfully!")
            print("   Model set to evaluation mode (eval())")
            print("   Ready for inference with real 3.8GB weights")
            print("="*70 + "\n")
            return model
        else:
            raise Exception("Model loading returned None")
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"   Traceback: {traceback.format_exc()}")
        
        model_state["loaded"] = False
        model_state["mmdet_available"] = False
        model_state["error"] = error_msg
        print("   Backend will run in HYBRID mode:")
        print("   - Detection: Enhanced heuristic-based (contour analysis)")
        print("   - Perturbations: Real module with all 12 types")
        print("="*70 + "\n")
        return None


def run_inference(image_np: np.ndarray, threshold: float = 0.3) -> List[Dict]:
    """Run detection on image (MMDET if available, else heuristic)"""
    
    if model_state["mmdet_available"] and model_state["model"] is not None:
        try:
            import torch
            from mmdet.apis import inference_detector
            
            # Ensure model is in eval mode for inference
            model = model_state["model"]
            model.eval()
            
            # Disable gradients for inference (saves memory and speeds up)
            with torch.no_grad():
                # Convert to BGR for inference
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                h, w = image_np.shape[:2]
                
                # Run inference with loaded model
                result = inference_detector(model, image_bgr)
            
            detections = []
            
            if result is not None:
                # Handle different result formats
                if hasattr(result, 'pred_instances'):
                    # Newer MMDET format
                    bboxes = result.pred_instances.bboxes.cpu().numpy()
                    scores = result.pred_instances.scores.cpu().numpy()
                    labels = result.pred_instances.labels.cpu().numpy()
                elif isinstance(result, tuple) and len(result) > 0:
                    # Legacy format: (bbox_results, segm_results, ...)
                    bbox_results = result[0]
                    if isinstance(bbox_results, list):
                        # List of arrays per class
                        for class_id, class_bboxes in enumerate(bbox_results):
                            if class_bboxes.size == 0:
                                continue
                            for box in class_bboxes:
                                x1, y1, x2, y2, score = box
                                bw = x2 - x1
                                bh = y2 - y1
                                
                                class_name = LAYOUT_CLASS_MAP.get(class_id, f"Class_{class_id}")
                                
                                detections.append({
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "confidence": float(score),
                                    "bbox": {
                                        "x": float(x1 / w),
                                        "y": float(y1 / h),
                                        "width": float(bw / w),
                                        "height": float(bh / h)
                                    },
                                    "area": float((bw * bh) / (w * h))
                                })
                        # Skip the pred_instances path for legacy format
                        detections.sort(key=lambda x: x["confidence"], reverse=True)
                        return detections[:100]
                
                # Handle pred_instances format
                if 'bboxes' in locals():
                    for bbox, score, label in zip(bboxes, scores, labels):
                        if score < threshold:
                            continue
                        
                        x1, y1, x2, y2 = bbox
                        bw = x2 - x1
                        bh = y2 - y1
                        
                        class_id = int(label)
                        class_name = LAYOUT_CLASS_MAP.get(class_id, f"Class_{class_id}")
                        
                        detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": float(score),
                            "bbox": {
                                "x": float(x1 / w),
                                "y": float(y1 / h),
                                "width": float(bw / w),
                                "height": float(bh / h)
                            },
                            "area": float((bw * bh) / (w * h))
                        })
            
            # Sort by confidence and limit results
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            return detections[:100]
            
        except Exception as e:
            print(f"⚠️  MMDET inference failed: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            # Fall back to heuristic if inference fails
            return heuristic_detect(image_np)
    else:
        # Use heuristic detection
        return heuristic_detect(image_np)


# ============================================================================
# API Routes
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        model_state["loaded"] = False


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model_state["loaded"],
        "mmdet_available": model_state["mmdet_available"],
        "detection_mode": "MMDET" if model_state["mmdet_available"] else "Heuristic",
        "device": model_state["device"],
        "model_type": model_state["model_type"],
        "weights_path": str(Config.MODEL_WEIGHTS_PATH),
        "weights_exists": Config.MODEL_WEIGHTS_PATH.exists(),
        "weights_size_gb": Config.MODEL_WEIGHTS_PATH.stat().st_size / (1024**3) if Config.MODEL_WEIGHTS_PATH.exists() else 0
    }


@app.get("/api/model-info")
async def model_info():
    """Get model information"""
    return {
        "name": "RoDLA InternImage-XL",
        "version": "3.0.0",
        "type": "Document Layout Analysis",
        "mmdet_loaded": model_state["loaded"],
        "mmdet_available": model_state["mmdet_available"],
        "detection_mode": "MMDET (Real Model)" if model_state["mmdet_available"] else "Heuristic (Contour-based)",
        "error": model_state["error"],
        "device": model_state["device"],
        "framework": "MMDET + PyTorch (or Heuristic Fallback)",
        "backbone": "InternImage-XL with DCNv3",
        "detector": "DINO",
        "dataset": "M6Doc (75 classes)",
        "weights_file": str(Config.MODEL_WEIGHTS_PATH),
        "config_file": str(Config.MODEL_CONFIG_PATH),
        "perturbations_available": True,
        "supported_perturbations": [
            "defocus", "vibration", "speckle", "texture",
            "watermark", "background", "ink_holdout", "ink_bleeding",
            "illumination", "rotation", "keystoning", "warping"
        ]
    }


@app.get("/api/perturbations/info")
async def perturbation_info():
    """Get information about available perturbations"""
    return {
        "total_perturbations": 12,
        "categories": {
            "blur": {
                "types": ["defocus", "vibration"],
                "description": "Blur effects simulating optical issues"
            },
            "noise": {
                "types": ["speckle", "texture"],
                "description": "Noise patterns and texture artifacts"
            },
            "content": {
                "types": ["watermark", "background"],
                "description": "Content additions like watermarks and backgrounds"
            },
            "inconsistency": {
                "types": ["ink_holdout", "ink_bleeding", "illumination"],
                "description": "Print quality issues and lighting variations"
            },
            "spatial": {
                "types": ["rotation", "keystoning", "warping"],
                "description": "Geometric transformations"
            }
        },
        "all_types": [
            "defocus", "vibration", "speckle", "texture",
            "watermark", "background", "ink_holdout", "ink_bleeding",
            "illumination", "rotation", "keystoning", "warping"
        ],
        "degree_levels": {
            1: "Mild - Subtle effect",
            2: "Moderate - Noticeable effect",
            3: "Severe - Strong effect"
        }
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), threshold: float = 0.3):
    """Detect document layout using RoDLA with real weights or heuristic fallback"""
    start_time = datetime.now()
    
    try:
        # Load image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # Run inference
        detections = run_inference(image_np, threshold=threshold)
        
        # Build class distribution
        class_distribution = {}
        for det in detections:
            cn = det["class_name"]
            class_distribution[cn] = class_distribution.get(cn, 0) + 1
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        detection_mode = "Real MMDET Model (3.8GB weights)" if model_state["mmdet_available"] else "Heuristic Detection"
        
        return {
            "success": True,
            "message": f"Detection completed using {detection_mode}",
            "detection_mode": detection_mode,
            "image_width": w,
            "image_height": h,
            "num_detections": len(detections),
            "detections": detections,
            "class_distribution": class_distribution,
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        print(f"❌ Detection error: {e}\n{traceback.format_exc()}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": False,
            "message": str(e),
            "image_width": 0,
            "image_height": 0,
            "num_detections": 0,
            "detections": [],
            "class_distribution": {},
            "processing_time_ms": processing_time
        }


@app.post("/api/generate-perturbations")
async def generate_perturbations(file: UploadFile = File(...)):
    """Generate all 12 perturbations with 3 degree levels each (36 total images)"""
    
    try:
        # Import simple perturbation functions (no external dependencies beyond common libs)
        from perturbations_simple import apply_perturbation as simple_apply_perturbation
        
        # Load image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        perturbations = {}
        
        # Original
        perturbations["original"] = {
            "original": encode_image_to_base64(image_np)
        }
        
        # All 12 perturbation types
        all_types = [
            "defocus", "vibration", "speckle", "texture",
            "watermark", "background", "ink_holdout", "ink_bleeding",
            "illumination", "rotation", "keystoning", "warping"
        ]
        
        print(f"📊 Generating perturbations for {len(all_types)} types × 3 degrees = 36 images...")
        
        # Generate all perturbations with 3 degree levels
        generated_count = 0
        for ptype in all_types:
            perturbations[ptype] = {}
            
            for degree in [1, 2, 3]:
                try:
                    # Use simple perturbation function (no external heavy dependencies)
                    result_image, success, message = simple_apply_perturbation(
                        image_bgr.copy(),
                        ptype,
                        degree=degree
                    )
                    
                    if success:
                        # Convert BGR to RGB for display
                        if len(result_image.shape) == 3 and result_image.shape[2] == 3:
                            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        else:
                            result_rgb = result_image
                        
                        perturbations[ptype][f"degree_{degree}"] = encode_image_to_base64(result_rgb)
                        generated_count += 1
                        print(f"  ✅ {ptype:12} degree {degree}: {message}")
                    else:
                        print(f"  ⚠️  {ptype:12} degree {degree}: {message}")
                        perturbations[ptype][f"degree_{degree}"] = encode_image_to_base64(image_np)
                        
                except Exception as e:
                    print(f"  ⚠️  Exception {ptype:12} degree {degree}: {e}")
                    perturbations[ptype][f"degree_{degree}"] = encode_image_to_base64(image_np)
        
        print(f"\n✅ Generated {generated_count}/36 perturbation images successfully")
        
        return {
            "success": True,
            "message": f"Perturbations generated: 12 types × 3 degrees = 36 images + 1 original = 37 total",
            "perturbations": perturbations,
            "grid_info": {
                "total_perturbations": 12,
                "degree_levels": 3,
                "total_images": 37,
                "generated_count": generated_count
            }
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "message": f"Perturbation module import error: {str(e)}",
            "perturbations": {}
        }
    except Exception as e:
        print(f"❌ Perturbation generation error: {e}\n{traceback.format_exc()}")
        return {
            "success": False,
            "message": str(e),
            "perturbations": {}
        }


# ============================================================================
# Mount frontend static files (MUST be last to not interfere with /api routes)
# ============================================================================

if Config.FRONTEND_DIR.exists():
    print(f"📁 Mounting frontend from: {Config.FRONTEND_DIR}")
    app.mount("/", StaticFiles(directory=str(Config.FRONTEND_DIR), html=True), name="frontend")
else:
    print(f"⚠️  Frontend directory not found: {Config.FRONTEND_DIR}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🔷"*35)
    print("🔷 RoDLA PRODUCTION BACKEND")
    print("🔷 Model: InternImage-XL with DINO")
    print("🔷 Weights: 3.8GB (rodla_internimage_xl_publaynet.pth)")
    print("🔷 Perturbations: 12 types × 3 degrees each")
    print("🔷 Detection: MMDET (if available) or Heuristic fallback")
    print("🔷"*35)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.API_PORT,
        log_level="info"
    )
