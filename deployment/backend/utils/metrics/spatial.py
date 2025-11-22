"""Spatial distribution analysis metrics"""
from typing import List, Dict
import numpy as np


def calculate_spatial_analysis(
    detections: List[Dict], 
    img_width: int, 
    img_height: int
) -> Dict:
    """
    Calculate comprehensive spatial distribution analysis
    
    Args:
        detections: List of detection dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        Dictionary of spatial metrics
    """
    if not detections:
        return {}
    
    centers_x = [d['bbox']['center_x'] for d in detections]
    centers_y = [d['bbox']['center_y'] for d in detections]
    areas = [d['area'] for d in detections]
    image_area = img_width * img_height
    
    # Quadrant analysis
    quadrants = _calculate_quadrants(centers_x, centers_y, img_width, img_height)
    
    # Horizontal and vertical distributions
    h_dist = _calculate_1d_distribution(centers_x, img_width)
    v_dist = _calculate_1d_distribution(centers_y, img_height)
    
    # Size distribution
    size_dist = _calculate_size_distribution(areas, image_area)
    
    # Density metrics
    avg_nn_dist = calculate_avg_nn_distance(centers_x, centers_y)
    clustering_score = calculate_clustering_score(centers_x, centers_y)
    
    return {
        "horizontal_distribution": h_dist,
        "vertical_distribution": v_dist,
        "quadrant_distribution": quadrants,
        "size_distribution": size_dist,
        "density_metrics": {
            "average_nearest_neighbor_distance": avg_nn_dist,
            "spatial_clustering_score": clustering_score
        }
    }


def calculate_avg_nn_distance(xs: List[float], ys: List[float]) -> float:
    """
    Calculate average nearest neighbor distance
    
    Args:
        xs: X coordinates
        ys: Y coordinates
        
    Returns:
        Average nearest neighbor distance
    """
    if len(xs) < 2:
        return 0.0
    
    points = np.column_stack((xs, ys))
    distances = []
    
    for i, point in enumerate(points):
        other_points = np.delete(points, i, axis=0)
        dists = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
        distances.append(np.min(dists))
    
    return round(float(np.mean(distances)), 2)


def calculate_clustering_score(xs: List[float], ys: List[float]) -> float:
    """
    Calculate spatial clustering score based on distribution
    
    Args:
        xs: X coordinates
        ys: Y coordinates
        
    Returns:
        Clustering score (0-1, higher = more clustered)
    """
    if len(xs) < 3:
        return 0.0
    
    mean_x, mean_y = np.mean(xs), np.mean(ys)
    distances = [
        np.sqrt((x - mean_x)**2 + (y - mean_y)**2) 
        for x, y in zip(xs, ys)
    ]
    
    # Normalized score (0-1, higher = more clustered)
    mean_dist = np.mean(distances)
    if mean_dist > 0:
        score = 1 - min(np.std(distances) / mean_dist, 1)
    else:
        score = 0
    
    return round(float(score), 4)


def calculate_skewness(data: List[float]) -> float:
    """
    Calculate skewness of distribution
    
    Args:
        data: List of values
        
    Returns:
        Skewness value
    """
    if len(data) < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    return float(np.mean(((np.array(data) - mean) / std) ** 3))


def _calculate_quadrants(
    xs: List[float], 
    ys: List[float], 
    width: int, 
    height: int
) -> Dict[str, int]:
    """Calculate detection distribution across quadrants"""
    quadrants = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    mid_x, mid_y = width / 2, height / 2
    
    for x, y in zip(xs, ys):
        if x < mid_x and y < mid_y:
            quadrants["Q1"] += 1
        elif x >= mid_x and y < mid_y:
            quadrants["Q2"] += 1
        elif x < mid_x and y >= mid_y:
            quadrants["Q3"] += 1
        else:
            quadrants["Q4"] += 1
    
    return quadrants


def _calculate_1d_distribution(
    values: List[float], 
    dimension: int
) -> Dict:
    """Calculate 1D distribution statistics"""
    third = dimension / 3
    two_thirds = 2 * dimension / 3
    
    return {
        "mean": round(float(np.mean(values)), 2),
        "std": round(float(np.std(values)), 2),
        "skewness": round(calculate_skewness(values), 3),
        "left_third": sum(1 for v in values if v < third),
        "center_third": sum(1 for v in values if third <= v < two_thirds),
        "right_third": sum(1 for v in values if v >= two_thirds)
    }


def _calculate_size_distribution(
    areas: List[float], 
    image_area: int
) -> Dict[str, int]:
    """Calculate size distribution categories"""
    return {
        "tiny": sum(1 for a in areas if a < image_area * 0.005),
        "small": sum(1 for a in areas if image_area * 0.005 <= a < image_area * 0.02),
        "medium": sum(1 for a in areas if image_area * 0.02 <= a < image_area * 0.1),
        "large": sum(1 for a in areas if a >= image_area * 0.1)
    }