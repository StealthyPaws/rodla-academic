"""Result processing and aggregation service"""
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import json
import base64
import numpy as np

from config.settings import OUTPUT_DIR
from utils.metrics.core import (
    calculate_core_metrics,
    calculate_confidence_metrics,
    calculate_class_metrics
)
from utils.metrics.rodla import (
    calculate_rodla_metrics,
    calculate_robustness_indicators
)
from utils.metrics.spatial import calculate_spatial_analysis
from utils.metrics.quality import (
    calculate_quality_metrics,
    calculate_layout_complexity
)
from utils.serialization import convert_to_json_serializable
from services.visualization import generate_comprehensive_visualizations
from services.interpretation import generate_comprehensive_interpretation


def create_comprehensive_results(
    detections: List[Dict],
    img_width: int,
    img_height: int,
    filename: str,
    score_threshold: float,
    generate_viz: bool = True
) -> Dict:
    """
    Create comprehensive detection results with all metrics
    
    Args:
        detections: List of detection dictionaries
        img_width: Image width
        img_height: Image height
        filename: Original filename
        score_threshold: Confidence threshold used
        generate_viz: Whether to generate visualizations
        
    Returns:
        Complete results dictionary
    """
    # Calculate all metrics
    core_metrics = calculate_core_metrics(detections, img_width, img_height)
    rodla_metrics = calculate_rodla_metrics(detections, core_metrics)
    spatial_metrics = calculate_spatial_analysis(detections, img_width, img_height)
    class_metrics = calculate_class_metrics(detections)
    confidence_metrics = calculate_confidence_metrics(detections)
    robustness_indicators = calculate_robustness_indicators(detections, core_metrics)
    layout_complexity = calculate_layout_complexity(detections, img_width, img_height)
    quality_metrics = calculate_quality_metrics(detections, img_width, img_height)
    
    # Generate visualizations if requested
    visualizations = {}
    if generate_viz:
        visualizations = generate_comprehensive_visualizations(
            detections, class_metrics, confidence_metrics,
            spatial_metrics, img_width, img_height
        )
    
    # Generate interpretation
    interpretation = generate_comprehensive_interpretation(
        core_metrics, rodla_metrics, class_metrics,
        layout_complexity, robustness_indicators
    )
    
    # Assemble complete results
    results = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
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
        "core_results": {
            "summary": core_metrics["summary"],
            "detections": detections[:20]  # Top 20 by confidence
        },
        "rodla_metrics": rodla_metrics,
        "spatial_analysis": spatial_metrics,
        "class_analysis": class_metrics,
        "confidence_analysis": confidence_metrics,
        "robustness_indicators": robustness_indicators,
        "layout_complexity": layout_complexity,
        "quality_metrics": quality_metrics,
        "visualizations": visualizations,
        "interpretation": interpretation,
        "all_detections": detections
    }
    
    return results


def save_results(
    results: Dict,
    filename: str,
    save_visualizations: bool = True
) -> Path:
    """
    Save results to JSON and visualization images
    
    Args:
        results: Complete results dictionary
        filename: Original filename
        save_visualizations: Whether to save visualization images
        
    Returns:
        Path to saved JSON file
    """
    safe_filename = Path(filename).stem
    json_path = OUTPUT_DIR / f"rodla_results_{safe_filename}.json"
    
    # Prepare JSON (remove base64 images to reduce file size)
    json_results = {k: v for k, v in results.items() if k != "visualizations"}
    json_results["visualizations_note"] = "Visualizations saved as separate PNG files"
    
    # Convert numpy types to JSON-serializable types
    json_results = convert_to_json_serializable(json_results)
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Saved JSON results to {json_path}")
    
    # Save visualizations as separate files
    if save_visualizations and results.get("visualizations"):
        for viz_name, viz_data in results["visualizations"].items():
            if viz_data:
                viz_path = OUTPUT_DIR / f"{safe_filename}_{viz_name}.png"
                # Decode base64 and save
                img_data = base64.b64decode(viz_data.split(',')[1])
                with open(viz_path, 'wb') as f:
                    f.write(img_data)
                print(f"✅ Saved visualization: {viz_path}")
    
    return json_path