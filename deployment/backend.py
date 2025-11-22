from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import tempfile
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

REPO_ROOT = Path("/mnt/d/MyStuff/University/Current/CV/Project/RoDLA")
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "model"))
sys.path.append(str(REPO_ROOT / "model/ops_dcnv3"))

from model.mmdet_custom.models.detectors import dino
from model.mmdet_custom.models.dense_heads import dino_head
from mmdet.apis import init_detector, inference_detector
import mmcv

app = FastAPI(title="RoDLA Object Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Baseline clean performance for mRD calculation (from paper)
BASELINE_MAP = {"M6Doc": 70.0, "PubLayNet": 96.0, "DocLayNet": 80.5}

@app.on_event("startup")
async def startup_event():
    global model
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("🔄 Loading RoDLA model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = init_detector(
            str(REPO_ROOT / "model/configs/m6doc/rodla_internimage_xl_m6doc.py"),
            str(REPO_ROOT / "rodla_internimage_xl_m6doc.pth"),
            device=device
        )
        print("✅ RoDLA Model loaded successfully")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise e


@app.get("/api/model-info")
async def get_model_info():
    if model is None:
        raise HTTPException(500, "Model not loaded")
    
    return JSONResponse({
        "model_name": "RoDLA InternImage-XL",
        "paper": "RoDLA: Benchmarking the Robustness of Document Layout Analysis Models (CVPR 2024)",
        "num_classes": len(model.CLASSES),
        "classes": model.CLASSES,
        "backbone": "InternImage-XL",
        "detection_framework": "DINO with Channel Attention + Average Pooling",
        "dataset": "M6Doc-P",
        "max_detections_per_image": 300,
        "state_of_the_art_performance": {
            "clean_mAP": 70.0,
            "perturbed_avg_mAP": 61.7,
            "mRD_score": 147.6
        }
    })


@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    score_thr: Optional[str] = Form("0.3"),
    return_image: Optional[str] = Form("false"),
    save_json: Optional[str] = Form("true"),
    generate_visualizations: Optional[str] = Form("true")
):
    """
    Comprehensive document layout detection with RoDLA metrics
    """
    tmp_path = None
    
    try:
        score_threshold = float(score_thr)
        should_return_image = return_image.lower() in ('true', '1', 'yes')
        should_save_json = save_json.lower() in ('true', '1', 'yes')
        should_generate_viz = generate_visualizations.lower() in ('true', '1', 'yes')
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Get image info
        img = Image.open(tmp_path)
        img_width, img_height = img.size
        
        # Run detection
        result = inference_detector(model, tmp_path)
        
        # Process detections
        detections = process_detections(result, score_threshold)
        
        # ========== COMPREHENSIVE ANALYSIS ==========
        
        # 1. Core Metrics
        core_metrics = calculate_core_metrics(detections, img_width, img_height)
        
        # 2. RoDLA-Specific Metrics (mPE and mRD estimation)
        rodla_metrics = calculate_rodla_metrics(detections, core_metrics)
        
        # 3. Spatial Analysis
        spatial_metrics = calculate_spatial_analysis(detections, img_width, img_height)
        
        # 4. Class-Specific Analysis
        class_metrics = calculate_class_metrics(detections)
        
        # 5. Confidence Distribution Analysis
        confidence_metrics = calculate_confidence_metrics(detections)
        
        # 6. Robustness Indicators
        robustness_indicators = calculate_robustness_indicators(detections, core_metrics)
        
        # 7. Layout Complexity Analysis
        layout_complexity = calculate_layout_complexity(detections, img_width, img_height)
        
        # 8. Detection Quality Metrics
        quality_metrics = calculate_quality_metrics(detections, img_width, img_height)
        
        # Generate visualizations
        visualizations = {}
        if should_generate_viz:
            visualizations = generate_comprehensive_visualizations(
                detections, class_metrics, confidence_metrics, 
                spatial_metrics, img_width, img_height
            )
        
        # Create comprehensive results
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "image_info": {
                "width": img_width,
                "height": img_height,
                "aspect_ratio": round(img_width / img_height, 3),
                "total_pixels": img_width * img_height
            },
            "detection_config": {
                "score_threshold": score_threshold,
                "model": "RoDLA InternImage-XL",
                "framework": "DINO with Robustness Enhancement",
                "max_detections": 300
            },
            
            # SECTION 1: Core Detection Results
            "core_results": {
                "summary": core_metrics["summary"],
                "detections": detections[:20]  # Top 20 by confidence
            },
            
            # SECTION 2: RoDLA-Specific Metrics
            "rodla_metrics": rodla_metrics,
            
            # SECTION 3: Spatial Analysis
            "spatial_analysis": spatial_metrics,
            
            # SECTION 4: Class Distribution & Statistics
            "class_analysis": class_metrics,
            
            # SECTION 5: Confidence Analysis
            "confidence_analysis": confidence_metrics,
            
            # SECTION 6: Robustness Indicators
            "robustness_indicators": robustness_indicators,
            
            # SECTION 7: Layout Complexity
            "layout_complexity": layout_complexity,
            
            # SECTION 8: Detection Quality
            "quality_metrics": quality_metrics,
            
            # SECTION 9: Visualizations (base64 encoded)
            "visualizations": visualizations,
            
            # SECTION 10: Human-Readable Interpretation
            "interpretation": generate_comprehensive_interpretation(
                core_metrics, rodla_metrics, class_metrics, 
                layout_complexity, robustness_indicators
            ),
            
            # Full detections list
            "all_detections": detections
        }
        
        # Save comprehensive JSON
        if should_save_json or not should_return_image:
            safe_filename = Path(file.filename).stem
            json_path = OUTPUT_DIR / f"rodla_results_{safe_filename}.json"
            
            # Remove base64 images from JSON to keep file size reasonable
            json_results = {k: v for k, v in results.items() if k != "visualizations"}
            json_results["visualizations_note"] = "Visualizations saved as separate PNG files"
            
            # Convert all numpy types to native Python types
            json_results = convert_to_json_serializable(json_results)
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"✅ Saved JSON results to {json_path}")
            
            # Save visualizations as separate files
            if visualizations:
                for viz_name, viz_data in visualizations.items():
                    if viz_data:
                        viz_path = OUTPUT_DIR / f"{safe_filename}_{viz_name}.png"
                        # Decode base64 and save
                        img_data = base64.b64decode(viz_data.split(',')[1])
                        with open(viz_path, 'wb') as f:
                            f.write(img_data)
        
        # Return annotated image if requested
        if should_return_image:
            safe_filename = Path(file.filename).name
            output_path = OUTPUT_DIR / f"annotated_{safe_filename}"
            
            model.show_result(
                tmp_path,
                result,
                score_thr=score_threshold,
                show=False,
                out_file=str(output_path)
            )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise HTTPException(500, "Failed to generate annotated image")

            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return FileResponse(
                path=str(output_path),
                media_type="image/jpeg",
                filename=f"annotated_{safe_filename}"
            )
        
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        return JSONResponse(results)
        
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


def process_detections(result, score_thr=0.3):
    """Convert detection results to detailed JSON format"""
    detections = []
    for class_id, class_result in enumerate(result):
        for bbox in class_result:
            if bbox[4] > score_thr:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    'class_id': int(class_id),
                    'class_name': model.CLASSES[class_id],
                    'bbox': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': width, 'height': height,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2
                    },
                    'confidence': float(bbox[4]),
                    'area': float(width * height),
                    'aspect_ratio': float(width / height) if height > 0 else 0
                })
    
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections


def calculate_core_metrics(detections: List[dict], img_width: int, img_height: int):
    """Core detection metrics"""
    if not detections:
        return {"summary": {"total_detections": 0}}
    
    total_area = sum(d['area'] for d in detections)
    image_area = img_width * img_height
    
    return {
        "summary": {
            "total_detections": len(detections),
            "unique_classes": len(set(d['class_name'] for d in detections)),
            "average_confidence": round(np.mean([d['confidence'] for d in detections]), 4),
            "median_confidence": round(np.median([d['confidence'] for d in detections]), 4),
            "min_confidence": round(min(d['confidence'] for d in detections), 4),
            "max_confidence": round(max(d['confidence'] for d in detections), 4),
            "coverage_percentage": round((total_area / image_area) * 100, 2),
            "average_detection_area": round(total_area / len(detections), 2)
        }
    }


def calculate_rodla_metrics(detections: List[dict], core_metrics: Dict):
    """
    Calculate RoDLA-specific metrics (mPE and mRD estimation)
    Note: Full mPE/mRD requires clean baseline and perturbation assessment
    Here we provide proxy metrics based on current detection
    """
    if not detections:
        return {}
    
    avg_conf = core_metrics["summary"]["average_confidence"]
    
    # Estimate perturbation effect based on confidence distribution
    # Lower and more varied confidence suggests potential perturbation
    conf_values = [d['confidence'] for d in detections]
    conf_std = np.std(conf_values)
    conf_range = max(conf_values) - min(conf_values)
    
    # Proxy mPE: Higher std and range suggest more perturbation effect
    estimated_mPE = round((conf_std * 100) + (conf_range * 50), 2)
    
    # Proxy mRD: Based on deviation from expected performance
    # mRD = (Degradation / mPE) * 100
    # Lower average confidence suggests more degradation
    expected_performance = 0.85  # Typical clean performance
    degradation = max(0, (expected_performance - avg_conf) * 100)
    
    estimated_mRD = round((degradation / max(estimated_mPE, 1)) * 100, 2) if estimated_mPE > 0 else 0
    
    return {
        "note": "These are estimated metrics. Full mRD/mPE require clean baseline comparison.",
        "estimated_mPE": estimated_mPE,
        "estimated_mRD": estimated_mRD,
        "confidence_std": round(conf_std, 4),
        "confidence_range": round(conf_range, 4),
        "robustness_score": round((1 - (estimated_mRD / 200)) * 100, 2),  # 0-100 scale
        "interpretation": {
            "mPE_level": "low" if estimated_mPE < 20 else "medium" if estimated_mPE < 40 else "high",
            "mRD_level": "excellent" if estimated_mRD < 100 else "good" if estimated_mRD < 150 else "needs_improvement",
            "overall_robustness": "high" if avg_conf > 0.8 else "medium" if avg_conf > 0.6 else "low"
        }
    }


def calculate_spatial_analysis(detections: List[dict], img_width: int, img_height: int):
    """Comprehensive spatial distribution analysis"""
    if not detections:
        return {}
    
    centers_x = [d['bbox']['center_x'] for d in detections]
    centers_y = [d['bbox']['center_y'] for d in detections]
    areas = [d['area'] for d in detections]
    image_area = img_width * img_height
    
    # Quadrant analysis
    quadrants = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    mid_x, mid_y = img_width / 2, img_height / 2
    
    for x, y in zip(centers_x, centers_y):
        if x < mid_x and y < mid_y:
            quadrants["Q1"] += 1
        elif x >= mid_x and y < mid_y:
            quadrants["Q2"] += 1
        elif x < mid_x and y >= mid_y:
            quadrants["Q3"] += 1
        else:
            quadrants["Q4"] += 1
    
    return {
        "horizontal_distribution": {
            "mean": round(np.mean(centers_x), 2),
            "std": round(np.std(centers_x), 2),
            "skewness": round(calculate_skewness(centers_x), 3),
            "left_third": sum(1 for x in centers_x if x < img_width / 3),
            "center_third": sum(1 for x in centers_x if img_width / 3 <= x < 2 * img_width / 3),
            "right_third": sum(1 for x in centers_x if x >= 2 * img_width / 3)
        },
        "vertical_distribution": {
            "mean": round(np.mean(centers_y), 2),
            "std": round(np.std(centers_y), 2),
            "skewness": round(calculate_skewness(centers_y), 3),
            "top_third": sum(1 for y in centers_y if y < img_height / 3),
            "middle_third": sum(1 for y in centers_y if img_height / 3 <= y < 2 * img_height / 3),
            "bottom_third": sum(1 for y in centers_y if y >= 2 * img_height / 3)
        },
        "quadrant_distribution": quadrants,
        "size_distribution": {
            "tiny": sum(1 for a in areas if a < image_area * 0.005),
            "small": sum(1 for a in areas if image_area * 0.005 <= a < image_area * 0.02),
            "medium": sum(1 for a in areas if image_area * 0.02 <= a < image_area * 0.1),
            "large": sum(1 for a in areas if a >= image_area * 0.1)
        },
        "density_metrics": {
            "average_nearest_neighbor_distance": calculate_avg_nn_distance(centers_x, centers_y),
            "spatial_clustering_score": calculate_clustering_score(centers_x, centers_y)
        }
    }


def calculate_class_metrics(detections: List[dict]):
    """Detailed per-class analysis"""
    if not detections:
        return {}
    
    class_data = {}
    for det in detections:
        cls = det['class_name']
        if cls not in class_data:
            class_data[cls] = {
                'count': 0,
                'confidences': [],
                'areas': [],
                'aspect_ratios': [],
                'positions': []
            }
        
        class_data[cls]['count'] += 1
        class_data[cls]['confidences'].append(det['confidence'])
        class_data[cls]['areas'].append(det['area'])
        class_data[cls]['aspect_ratios'].append(det['aspect_ratio'])
        class_data[cls]['positions'].append((det['bbox']['center_x'], det['bbox']['center_y']))
    
    class_metrics = {}
    total_detections = len(detections)
    
    for cls, data in class_data.items():
        class_metrics[cls] = {
            "count": data['count'],
            "percentage": round((data['count'] / total_detections) * 100, 2),
            "confidence_stats": {
                "mean": round(np.mean(data['confidences']), 4),
                "std": round(np.std(data['confidences']), 4),
                "min": round(min(data['confidences']), 4),
                "max": round(max(data['confidences']), 4)
            },
            "area_stats": {
                "mean": round(np.mean(data['areas']), 2),
                "std": round(np.std(data['areas']), 2),
                "total": round(sum(data['areas']), 2)
            },
            "aspect_ratio_stats": {
                "mean": round(np.mean(data['aspect_ratios']), 3),
                "orientation": "horizontal" if np.mean(data['aspect_ratios']) > 1.2 else "vertical" if np.mean(data['aspect_ratios']) < 0.8 else "square"
            }
        }
    
    return class_metrics


def calculate_confidence_metrics(detections: List[dict]):
    """Detailed confidence distribution analysis"""
    if not detections:
        return {}
    
    confidences = [d['confidence'] for d in detections]
    
    # Bin confidence into ranges
    bins = {
        "very_high (0.9-1.0)": sum(1 for c in confidences if c >= 0.9),
        "high (0.8-0.9)": sum(1 for c in confidences if 0.8 <= c < 0.9),
        "medium (0.6-0.8)": sum(1 for c in confidences if 0.6 <= c < 0.8),
        "low (0.4-0.6)": sum(1 for c in confidences if 0.4 <= c < 0.6),
        "very_low (0-0.4)": sum(1 for c in confidences if c < 0.4)
    }
    
    return {
        "distribution": {
            "mean": round(np.mean(confidences), 4),
            "median": round(np.median(confidences), 4),
            "std": round(np.std(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "q1": round(np.percentile(confidences, 25), 4),
            "q3": round(np.percentile(confidences, 75), 4)
        },
        "binned_distribution": bins,
        "percentages": {k: round((v / len(detections)) * 100, 2) for k, v in bins.items()},
        "entropy": round(calculate_entropy(confidences), 4)
    }


def calculate_robustness_indicators(detections: List[dict], core_metrics: Dict):
    """Indicators of model robustness and detection stability"""
    if not detections:
        return {}
    
    confidences = [d['confidence'] for d in detections]
    avg_conf = core_metrics["summary"]["average_confidence"]
    
    # Coefficient of variation (lower = more stable)
    cv = (np.std(confidences) / avg_conf) if avg_conf > 0 else 0
    
    # Detection consistency score
    high_conf_ratio = sum(1 for c in confidences if c >= 0.8) / len(confidences)
    
    return {
        "stability_score": round((1 - cv) * 100, 2),
        "coefficient_of_variation": round(cv, 4),
        "high_confidence_ratio": round(high_conf_ratio, 4),
        "prediction_consistency": "high" if cv < 0.15 else "medium" if cv < 0.3 else "low",
        "model_certainty": "high" if avg_conf > 0.8 else "medium" if avg_conf > 0.6 else "low",
        "robustness_rating": calculate_robustness_rating(avg_conf, cv, high_conf_ratio)
    }


def calculate_layout_complexity(detections: List[dict], img_width: int, img_height: int):
    """Analyze document layout complexity"""
    if not detections:
        return {}
    
    unique_classes = len(set(d['class_name'] for d in detections))
    total_detections = len(detections)
    
    # Calculate spatial complexity
    centers = [(d['bbox']['center_x'], d['bbox']['center_y']) for d in detections]
    avg_distance = calculate_avg_nn_distance(
        [c[0] for c in centers],
        [c[1] for c in centers]
    )
    
    # Normalized metrics
    detection_density = total_detections / (img_width * img_height) * 1000000
    
    complexity_score = (
        (unique_classes / 20) * 30 +  # Class diversity (max 20 classes)
        (min(total_detections / 50, 1)) * 30 +  # Detection count
        (min(detection_density / 10, 1)) * 20 +  # Density
        (1 - min(avg_distance / 500, 1)) * 20  # Spatial clustering
    )
    
    return {
        "class_diversity": int(unique_classes),
        "total_elements": int(total_detections),
        "detection_density": round(float(detection_density), 4),
        "average_element_distance": round(float(avg_distance), 2),
        "complexity_score": round(float(complexity_score), 2),
        "complexity_level": "simple" if complexity_score < 30 else "moderate" if complexity_score < 60 else "complex",
        "layout_characteristics": {
            "is_dense": bool(detection_density > 5),
            "is_diverse": bool(unique_classes >= 10),
            "is_structured": bool(avg_distance < 200)
        }
    }


def calculate_quality_metrics(detections: List[dict], img_width: int, img_height: int):
    """Detection quality assessment metrics"""
    if not detections:
        return {}
    
    # Overlap analysis
    overlaps = calculate_detection_overlaps(detections)
    
    # Size consistency
    areas = [d['area'] for d in detections]
    size_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
    
    return {
        "overlap_analysis": {
            "total_overlapping_pairs": overlaps['count'],
            "overlap_percentage": round(overlaps['percentage'], 2),
            "average_iou": round(overlaps['avg_iou'], 4)
        },
        "size_consistency": {
            "coefficient_of_variation": round(size_cv, 4),
            "consistency_level": "high" if size_cv < 0.5 else "medium" if size_cv < 1.0 else "low"
        },
        "detection_quality_score": round((1 - min(overlaps['percentage'] / 100, 1)) * 50 + 
                                        (1 - min(size_cv, 1)) * 50, 2)
    }


# ========== HELPER FUNCTIONS ==========

def calculate_skewness(data):
    """Calculate skewness of distribution"""
    if len(data) < 3:
        return 0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)


def calculate_avg_nn_distance(xs, ys):
    """Calculate average nearest neighbor distance"""
    if len(xs) < 2:
        return 0
    
    points = np.column_stack((xs, ys))
    distances = []
    
    for i, point in enumerate(points):
        other_points = np.delete(points, i, axis=0)
        dists = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
        distances.append(np.min(dists))
    
    return round(np.mean(distances), 2)


def calculate_clustering_score(xs, ys):
    """Simple clustering score based on spatial distribution"""
    if len(xs) < 3:
        return 0
    
    # Use std of distances as clustering indicator
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    distances = [np.sqrt((x - mean_x)**2 + (y - mean_y)**2) for x, y in zip(xs, ys)]
    
    # Normalized score (0-1, higher = more clustered)
    score = 1 - min(np.std(distances) / np.mean(distances), 1) if np.mean(distances) > 0 else 0
    return round(score, 4)


def calculate_entropy(values):
    """Calculate Shannon entropy of distribution"""
    hist, _ = np.histogram(values, bins=10)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs))


def calculate_robustness_rating(avg_conf, cv, high_conf_ratio):
    """Overall robustness rating"""
    score = (avg_conf * 40) + ((1 - cv) * 30) + (high_conf_ratio * 30)
    
    if score >= 80:
        return {"rating": "excellent", "score": round(score, 2)}
    elif score >= 60:
        return {"rating": "good", "score": round(score, 2)}
    elif score >= 40:
        return {"rating": "fair", "score": round(score, 2)}
    else:
        return {"rating": "poor", "score": round(score, 2)}


def calculate_detection_overlaps(detections):
    """Calculate IoU overlaps between detections"""
    overlaps = []
    
    for i, det1 in enumerate(detections):
        for det2 in detections[i+1:]:
            iou = calculate_iou(det1['bbox'], det2['bbox'])
            if iou > 0:
                overlaps.append(iou)
    
    return {
        'count': len(overlaps),
        'percentage': (len(overlaps) / max(len(detections), 1)) * 100,
        'avg_iou': np.mean(overlaps) if overlaps else 0
    }


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2'])
    y2 = min(bbox1['y2'], bbox2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def generate_comprehensive_visualizations(detections, class_metrics, confidence_metrics, 
                                          spatial_metrics, img_width, img_height):
    """Generate all visualizations as base64 encoded images"""
    visualizations = {}
    
    if not detections:
        return visualizations
    
    # 1. Class Distribution Bar Chart
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(class_metrics.keys())
        counts = [class_metrics[c]['count'] for c in classes]
        
        bars = ax.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        visualizations['class_distribution'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating class distribution: {e}")
    
    # 2. Confidence Distribution Histogram
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        confidences = [d['confidence'] for d in detections]
        
        ax.hist(confidences, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
        ax.axvline(np.median(confidences), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(confidences):.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        visualizations['confidence_distribution'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating confidence distribution: {e}")
    
    # 3. Spatial Distribution Heatmap
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        centers_x = [d['bbox']['center_x'] for d in detections]
        centers_y = [d['bbox']['center_y'] for d in detections]
        
        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(centers_x, centers_y, bins=20)
        
        im = ax.imshow(heatmap.T, origin='lower', cmap='YlOrRd', aspect='auto',
                      extent=[0, img_width, 0, img_height])
        
        ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
        ax.set_title('Spatial Distribution Heatmap', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Detection Density')
        plt.tight_layout()
        visualizations['spatial_heatmap'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating spatial heatmap: {e}")
    
    # 4. Box Plot: Confidence by Class
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class_conf_data = []
        class_labels = []
        
        for cls in sorted(class_metrics.keys(), key=lambda x: class_metrics[x]['count'], reverse=True)[:10]:
            conf_values = [d['confidence'] for d in detections if d['class_name'] == cls]
            class_conf_data.append(conf_values)
            class_labels.append(f"{cls}\n(n={len(conf_values)})")
        
        # Fix matplotlib deprecation warning
        bp = ax.boxplot(class_conf_data, tick_labels=class_labels, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Distribution by Class (Top 10)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        visualizations['confidence_by_class'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating confidence by class: {e}")
    
    # 5. Area Distribution Scatter
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        areas = [d['area'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        
        scatter = ax.scatter(areas, confidences, alpha=0.6, c=confidences, 
                           cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Detection Area (pixels²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Detection Area vs Confidence', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Confidence')
        plt.tight_layout()
        visualizations['area_vs_confidence'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating area vs confidence: {e}")
    
    # 6. Quadrant Distribution Pie Chart
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        quadrants = spatial_metrics['quadrant_distribution']
        labels = [f'{k}\n({v} elements)' for k, v in quadrants.items()]
        sizes = list(quadrants.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Detection Distribution by Quadrant', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        visualizations['quadrant_distribution'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating quadrant distribution: {e}")
    
    # 7. Size Distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        size_dist = spatial_metrics['size_distribution']
        categories = list(size_dist.keys())
        values = list(size_dist.values())
        
        bars = ax.bar(categories, values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
        ax.set_xlabel('Size Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Detection Size Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        visualizations['size_distribution'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating size distribution: {e}")
    
    # 8. Top Classes by Average Confidence
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sorted_classes = sorted(class_metrics.items(), 
                               key=lambda x: x[1]['confidence_stats']['mean'], 
                               reverse=True)[:15]
        
        classes = [c[0] for c in sorted_classes]
        avg_confs = [c[1]['confidence_stats']['mean'] for c in sorted_classes]
        
        bars = ax.barh(range(len(classes)), avg_confs, color='coral', alpha=0.8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_xlabel('Average Confidence', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Classes by Average Confidence', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{avg_confs[i]:.3f}', ha='left', va='center', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        visualizations['top_classes_confidence'] = fig_to_base64(fig)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating top classes: {e}")
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return f"data:image/png;base64,{image_base64}"


def generate_comprehensive_interpretation(core_metrics, rodla_metrics, class_metrics, 
                                         layout_complexity, robustness_indicators):
    """Generate detailed human-readable interpretation"""
    
    if not core_metrics.get("summary"):
        return {"overview": "No detections found in the document."}
    
    summary = core_metrics["summary"]
    
    # Overview
    overview = f"""Document Analysis Summary:
Detected {summary['total_detections']} layout elements across {summary['unique_classes']} different classes. 
The model achieved an average confidence of {summary['average_confidence']:.1%}, indicating 
{robustness_indicators['model_certainty']} certainty in predictions. 
The detected elements cover {summary['coverage_percentage']:.1f}% of the document area."""
    
    # Most common elements
    top_3_classes = sorted(class_metrics.items(), 
                          key=lambda x: x[1]['count'], 
                          reverse=True)[:3]
    
    top_elements_desc = "The most common elements are: " + ", ".join([
        f"{cls} ({data['count']} instances, {data['confidence_stats']['mean']:.1%} avg confidence)"
        for cls, data in top_3_classes
    ])
    
    # RoDLA-specific insights
    rodla_desc = f"""RoDLA Robustness Analysis:
Estimated perturbation effect (mPE): {rodla_metrics.get('estimated_mPE', 'N/A')} - {rodla_metrics.get('interpretation', {}).get('mPE_level', 'unknown')} level
Estimated robustness degradation (mRD): {rodla_metrics.get('estimated_mRD', 'N/A')} - {rodla_metrics.get('interpretation', {}).get('mRD_level', 'unknown')}
Overall robustness: {rodla_metrics.get('interpretation', {}).get('overall_robustness', 'unknown')}
Robustness score: {rodla_metrics.get('robustness_score', 'N/A')}/100"""
    
    # Layout complexity
    complexity_desc = f"""Layout Complexity:
Complexity level: {layout_complexity['complexity_level']} (score: {layout_complexity['complexity_score']:.1f}/100)
Class diversity: {layout_complexity['class_diversity']} unique element types
Detection density: {layout_complexity['detection_density']:.2f} elements per megapixel
Spatial structure: {'Structured' if layout_complexity['layout_characteristics']['is_structured'] else 'Unstructured'}"""
    
    # Key findings
    key_findings = []
    
    # Confidence findings
    if summary['average_confidence'] > 0.85:
        key_findings.append("✓ Excellent detection confidence - model is highly certain about predictions")
    elif summary['average_confidence'] < 0.6:
        key_findings.append("⚠ Lower confidence detected - document may have quality issues or unusual layout")
    
    # Coverage findings
    if summary['coverage_percentage'] > 70:
        key_findings.append("✓ High document coverage - most of the page contains layout elements")
    elif summary['coverage_percentage'] < 20:
        key_findings.append("⚠ Low document coverage - sparse layout with significant white space")
    
    # Robustness findings
    if robustness_indicators['stability_score'] > 80:
        key_findings.append("✓ High stability - consistent predictions across detections")
    elif robustness_indicators['stability_score'] < 50:
        key_findings.append("⚠ Variable predictions - consider potential document perturbations")
    
    # Complexity findings
    if layout_complexity['complexity_level'] == 'complex':
        key_findings.append("ℹ Complex document structure with diverse element types")
    elif layout_complexity['complexity_level'] == 'simple':
        key_findings.append("ℹ Simple document structure with limited element diversity")
    
    # Recommendations
    recommendations = []
    
    if summary['average_confidence'] < 0.7:
        recommendations.append("Consider pre-processing the image (denoising, contrast adjustment)")
    
    if rodla_metrics.get('estimated_mRD', 0) > 150:
        recommendations.append("High robustness degradation detected - document may have perturbations (blur, noise, distortions)")
    
    if layout_complexity['layout_characteristics']['is_dense'] and summary['average_confidence'] < 0.75:
        recommendations.append("Dense layout with lower confidence - verify detection accuracy manually")
    
    # Perturbation assessment
    perturbation_assessment = "Based on RoDLA metrics:\n"
    if rodla_metrics.get('estimated_mRD', 0) < 100:
        perturbation_assessment += "✓ Minimal to no perturbation effects detected\n"
    elif rodla_metrics.get('estimated_mRD', 0) < 150:
        perturbation_assessment += "⚠ Moderate perturbation effects may be present\n"
    else:
        perturbation_assessment += "⚠ Significant perturbation effects detected\n"
    
    perturbation_assessment += f"Confidence variability: {robustness_indicators['coefficient_of_variation']:.3f} "
    perturbation_assessment += f"({'low variability - stable' if robustness_indicators['coefficient_of_variation'] < 0.15 else 'high variability - potentially perturbed'})"
    
    return {
        "overview": overview,
        "top_elements": top_elements_desc,
        "rodla_analysis": rodla_desc,
        "layout_complexity": complexity_desc,
        "key_findings": key_findings,
        "perturbation_assessment": perturbation_assessment,
        "recommendations": recommendations if recommendations else ["No specific recommendations - detection quality is good"],
        "confidence_summary": {
            "level": robustness_indicators['model_certainty'],
            "stability": robustness_indicators['prediction_consistency'],
            "rating": robustness_indicators['robustness_rating']['rating']
        }
    }


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
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
    else:
        return obj


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)