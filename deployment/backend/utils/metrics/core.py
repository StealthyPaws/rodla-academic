"""Core detection metrics calculations"""
from typing import List, Dict
import numpy as np


def calculate_core_metrics(
    detections: List[Dict], 
    img_width: int, 
    img_height: int
) -> Dict:
    """
    Calculate core detection metrics
    
    Args:
        detections: List of detection dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        Dictionary of core metrics
    """
    if not detections:
        return {"summary": {"total_detections": 0}}
    
    total_area = sum(d['area'] for d in detections)
    image_area = img_width * img_height
    confidences = [d['confidence'] for d in detections]
    
    return {
        "summary": {
            "total_detections": len(detections),
            "unique_classes": len(set(d['class_name'] for d in detections)),
            "average_confidence": round(np.mean(confidences), 4),
            "median_confidence": round(np.median(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "coverage_percentage": round((total_area / image_area) * 100, 2),
            "average_detection_area": round(total_area / len(detections), 2)
        }
    }


def calculate_confidence_metrics(detections: List[Dict]) -> Dict:
    """
    Calculate detailed confidence distribution metrics
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary of confidence metrics
    """
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
        "percentages": {
            k: round((v / len(detections)) * 100, 2) 
            for k, v in bins.items()
        },
        "entropy": round(_calculate_entropy(confidences), 4)
    }


def calculate_class_metrics(detections: List[Dict]) -> Dict:
    """
    Calculate per-class analysis metrics
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary of class-specific metrics
    """
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
                'aspect_ratios': []
            }
        
        class_data[cls]['count'] += 1
        class_data[cls]['confidences'].append(det['confidence'])
        class_data[cls]['areas'].append(det['area'])
        class_data[cls]['aspect_ratios'].append(det['aspect_ratio'])
    
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
                "orientation": _get_orientation(np.mean(data['aspect_ratios']))
            }
        }
    
    return class_metrics


def _calculate_entropy(values: List[float]) -> float:
    """Calculate Shannon entropy of distribution"""
    hist, _ = np.histogram(values, bins=10)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs))


def _get_orientation(aspect_ratio: float) -> str:
    """Determine orientation from aspect ratio"""
    if aspect_ratio > 1.2:
        return "horizontal"
    elif aspect_ratio < 0.8:
        return "vertical"
    else:
        return "square"