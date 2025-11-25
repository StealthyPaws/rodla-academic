"""Detection quality and layout complexity metrics"""
from typing import List, Dict
import numpy as np


def calculate_quality_metrics(
    detections: List[Dict], 
    img_width: int, 
    img_height: int
) -> Dict:
    """
    Calculate detection quality assessment metrics
    
    Args:
        detections: List of detection dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        Dictionary of quality metrics
    """
    if not detections:
        return {}
    
    # Overlap analysis
    overlaps = calculate_detection_overlaps(detections)
    
    # Size consistency
    areas = [d['area'] for d in detections]
    mean_area = np.mean(areas)
    size_cv = (np.std(areas) / mean_area) if mean_area > 0 else 0
    
    # Overall quality score
    quality_score = (
        (1 - min(overlaps['percentage'] / 100, 1)) * 50 + 
        (1 - min(size_cv, 1)) * 50
    )
    
    return {
        "overlap_analysis": {
            "total_overlapping_pairs": overlaps['count'],
            "overlap_percentage": round(overlaps['percentage'], 2),
            "average_iou": round(overlaps['avg_iou'], 4)
        },
        "size_consistency": {
            "coefficient_of_variation": round(size_cv, 4),
            "consistency_level": _get_consistency_level(size_cv)
        },
        "detection_quality_score": round(quality_score, 2)
    }


def calculate_layout_complexity(
    detections: List[Dict], 
    img_width: int, 
    img_height: int
) -> Dict:
    """
    Analyze document layout complexity
    
    Args:
        detections: List of detection dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        Dictionary of complexity metrics
    """
    if not detections:
        return {}
    
    unique_classes = len(set(d['class_name'] for d in detections))
    total_detections = len(detections)
    
    # Calculate spatial complexity
    centers = [
        (d['bbox']['center_x'], d['bbox']['center_y']) 
        for d in detections
    ]
    avg_distance = _calculate_avg_distance(centers)
    
    # Normalized metrics
    detection_density = total_detections / (img_width * img_height) * 1000000
    
    # Complexity score (0-100)
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
        "complexity_level": _get_complexity_level(complexity_score),
        "layout_characteristics": {
            "is_dense": bool(detection_density > 5),
            "is_diverse": bool(unique_classes >= 10),
            "is_structured": bool(avg_distance < 200)
        }
    }


def calculate_detection_overlaps(detections: List[Dict]) -> Dict:
    """
    Calculate IoU overlaps between detections
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Overlap statistics
    """
    overlaps = []
    
    for i, det1 in enumerate(detections):
        for det2 in detections[i+1:]:
            iou = calculate_iou(det1['bbox'], det2['bbox'])
            if iou > 0:
                overlaps.append(iou)
    
    total_pairs = len(detections) if len(detections) > 0 else 1
    
    return {
        'count': len(overlaps),
        'percentage': (len(overlaps) / total_pairs) * 100,
        'avg_iou': float(np.mean(overlaps)) if overlaps else 0.0
    }


def calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union between two bounding boxes
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU value
    """
    x1 = max(bbox1['x1'], bbox2['x1'])
    y1 = max(bbox1['y1'], bbox2['y1'])
    x2 = min(bbox1['x2'], bbox2['x2'])
    y2 = min(bbox1['y2'], bbox2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    union = area1 + area2 - intersection
    
    return float(intersection / union) if union > 0 else 0.0


def _calculate_avg_distance(centers: List[tuple]) -> float:
    """Calculate average pairwise distance between centers"""
    if len(centers) < 2:
        return 0.0
    
    distances = []
    for i, (x1, y1) in enumerate(centers):
        for x2, y2 in centers[i+1:]:
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)
    
    return float(np.mean(distances)) if distances else 0.0


def _get_consistency_level(cv: float) -> str:
    """Categorize consistency level"""
    if cv < 0.5:
        return "high"
    elif cv < 1.0:
        return "medium"
    else:
        return "low"


def _get_complexity_level(score: float) -> str:
    """Categorize complexity level"""
    if score < 30:
        return "simple"
    elif score < 60:
        return "moderate"
    else:
        return "complex"