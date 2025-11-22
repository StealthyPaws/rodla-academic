"""
Perturbation Application Module
================================
Central logic for applying perturbations to images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from .blur import apply_defocus, apply_vibration
from .content import add_watermark, add_image_background
from .noise import apply_speckle, apply_texture
from .inconsistency import apply_ink_holdout, apply_ink_bleeding, apply_illumination
from .spatial import apply_rotation, apply_keystoning, apply_warping


# Perturbation categories
PERTURBATION_CATEGORIES = {
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
}

# Flatten for easy validation
ALL_PERTURBATIONS = [
    p for category in PERTURBATION_CATEGORIES.values() 
    for p in category["types"]
]


def apply_perturbation(
    image: np.ndarray,
    perturbation_type: str,
    degree: int = 1,
    background_folder: Optional[str] = None
) -> Tuple[np.ndarray, bool, str]:
    """
    Apply a single perturbation to an image.
    
    Args:
        image: Input image (BGR format from cv2)
        perturbation_type: Type of perturbation to apply
        degree: Intensity level (1=mild, 2=moderate, 3=severe)
        background_folder: Path to background images (only for 'background' type)
    
    Returns:
        Tuple of (perturbed_image, success_flag, message)
    
    Raises:
        ValueError: If perturbation_type is invalid or degree out of range
    """
    # Validation
    if perturbation_type not in ALL_PERTURBATIONS:
        raise ValueError(
            f"Invalid perturbation type: '{perturbation_type}'. "
            f"Must be one of: {', '.join(ALL_PERTURBATIONS)}"
        )
    
    if not 1 <= degree <= 3:
        raise ValueError(f"Degree must be 1, 2, or 3. Got: {degree}")
    
    try:
        result_image = None
        
        # Blur perturbations
        if perturbation_type == "defocus":
            result_image = apply_defocus(image, degree=degree)
            
        elif perturbation_type == "vibration":
            result_image = apply_vibration(image, degree=degree)
        
        # Noise perturbations
        elif perturbation_type == "speckle":
            result_image = apply_speckle(image, degree=degree)
            
        elif perturbation_type == "texture":
            result_image = apply_texture(image, degree=degree)
        
        # Content perturbations
        elif perturbation_type == "watermark":
            result_image = add_watermark(image, degree=degree)
            
        elif perturbation_type == "background":
            if background_folder is None:
                return image, False, "background_folder required for 'background' perturbation"
            result_image = add_image_background(image, degree=degree, background_folder=background_folder)
        
        # Inconsistency perturbations
        elif perturbation_type == "ink_holdout":
            result_image = apply_ink_holdout(image, degree=degree)
            
        elif perturbation_type == "ink_bleeding":
            result_image = apply_ink_bleeding(image, degree=degree)
            
        elif perturbation_type == "illumination":
            result_image = apply_illumination(image, degree=degree)
        
        # Spatial perturbations (these return tuples with annotations)
        elif perturbation_type == "rotation":
            result_image, _ = apply_rotation(image, annos=None, degree=degree)
            
        elif perturbation_type == "keystoning":
            result_image, _ = apply_keystoning(image, annos=None, degree=degree)
            
        elif perturbation_type == "warping":
            result_image, _ = apply_warping(image, annos=None, degree=degree)
        
        if result_image is None:
            return image, False, f"Perturbation '{perturbation_type}' returned None"
        
        return result_image, True, f"Successfully applied {perturbation_type} (degree {degree})"
    
    except Exception as e:
        error_msg = f"Error applying {perturbation_type}: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return image, False, error_msg


def apply_multiple_perturbations(
    image: np.ndarray,
    perturbations: List[Dict],
    background_folder: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply multiple perturbations sequentially.
    
    Args:
        image: Input image (BGR format)
        perturbations: List of dicts with keys 'type' and 'degree'
        background_folder: Optional background folder for 'background' perturbation
    
    Returns:
        Tuple of (final_perturbed_image, results_dict)
    
    Example:
        perturbations = [
            {"type": "defocus", "degree": 2},
            {"type": "speckle", "degree": 1},
            {"type": "rotation", "degree": 1}
        ]
        
        perturbed_img, results = apply_multiple_perturbations(img, perturbations)
    """
    current_image = image.copy()
    results = {
        "applied": [],
        "failed": [],
        "total": len(perturbations),
        "success_rate": 0.0
    }
    
    for idx, pert in enumerate(perturbations):
        pert_type = pert.get("type")
        degree = pert.get("degree", 1)
        
        # Use provided background_folder or from perturbation config
        bg_folder = pert.get("background_folder", background_folder)
        
        current_image, success, message = apply_perturbation(
            current_image,
            pert_type,
            degree,
            bg_folder
        )
        
        if success:
            results["applied"].append({
                "order": idx + 1,
                "type": pert_type,
                "degree": degree,
                "message": message
            })
        else:
            results["failed"].append({
                "order": idx + 1,
                "type": pert_type,
                "degree": degree,
                "error": message
            })
    
    # Calculate success rate
    if results["total"] > 0:
        results["success_rate"] = len(results["applied"]) / results["total"]
    
    return current_image, results


def get_perturbation_info() -> Dict:
    """
    Get information about available perturbations.
    
    Returns:
        Dictionary with perturbation categories and descriptions
    """
    return {
        "total_perturbations": len(ALL_PERTURBATIONS),
        "categories": PERTURBATION_CATEGORIES,
        "all_types": ALL_PERTURBATIONS,
        "degree_levels": {
            1: "Mild - Subtle effect",
            2: "Moderate - Noticeable effect",
            3: "Severe - Strong effect"
        },
        "notes": {
            "background": "Requires background_folder parameter",
            "spatial": "May change image dimensions slightly",
            "sequential": "Multiple perturbations applied in order specified"
        }
    }