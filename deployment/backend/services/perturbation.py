"""
Perturbation Service
====================
Business logic for applying image perturbations.
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from PIL import Image
from datetime import datetime

from perturbations import (
    apply_multiple_perturbations,
    get_perturbation_info,
    PERTURBATION_CATEGORIES
)
from config.settings import OUTPUT_DIR


def perturb_image_service(
    image: np.ndarray,
    perturbations: List[dict],
    filename: str,
    save_image: bool = False,
    return_base64: bool = False,
    background_folder: Optional[str] = None
) -> Dict:
    """
    Apply perturbations to an image and prepare response.
    
    Args:
        image: Input image (BGR from cv2)
        perturbations: List of perturbation configurations
        filename: Original filename
        save_image: Whether to save perturbed image to disk
        return_base64: Whether to return image as base64 string
        background_folder: Optional background folder path
    
    Returns:
        Dictionary with perturbation results
    """
    # Store original shape
    original_shape = list(image.shape)
    
    # Apply perturbations
    perturbed_image, results = apply_multiple_perturbations(
        image,
        perturbations,
        background_folder
    )
    
    # Store perturbed shape
    perturbed_shape = list(perturbed_image.shape)
    
    # Prepare response
    response = {
        "success": len(results["applied"]) > 0,
        "message": f"Applied {len(results['applied'])}/{results['total']} perturbations successfully",
        "perturbations_applied": results["applied"],
        "perturbations_failed": results["failed"],
        "total_perturbations": results["total"],
        "success_rate": results["success_rate"],
        "original_shape": original_shape,
        "perturbed_shape": perturbed_shape,
        "image_base64": None,
        "saved_path": None
    }
    
    # Convert to base64 if requested
    if return_base64:
        response["image_base64"] = image_to_base64(perturbed_image)
    
    # Save to disk if requested
    if save_image:
        saved_path = save_perturbed_image(perturbed_image, filename, perturbations)
        response["saved_path"] = str(saved_path)
    
    return response, perturbed_image


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV image to base64 string.
    
    Args:
        image: BGR image from cv2
    
    Returns:
        Base64 encoded string with data URI prefix
    """
    # Encode image to PNG
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Return with data URI prefix
    return f"data:image/png;base64,{image_base64}"


def save_perturbed_image(
    image: np.ndarray,
    original_filename: str,
    perturbations: List[dict]
) -> Path:
    """
    Save perturbed image to disk with descriptive filename.
    
    Args:
        image: Perturbed image
        original_filename: Original file name
        perturbations: List of applied perturbations
    
    Returns:
        Path to saved image
    """
    # Create perturbations subdirectory
    pert_dir = OUTPUT_DIR / "perturbations"
    pert_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(original_filename).stem
    
    # Create perturbation suffix
    pert_suffix = "_".join([
        f"{p['type']}{p['degree']}" 
        for p in perturbations
    ])[:50]  # Limit length
    
    filename = f"{base_name}_pert_{pert_suffix}_{timestamp}.png"
    save_path = pert_dir / filename
    
    # Save image
    cv2.imwrite(str(save_path), image)
    
    return save_path


def get_perturbation_info_service() -> Dict:
    """
    Get comprehensive perturbation information.
    
    Returns:
        Dictionary with perturbation details
    """
    return get_perturbation_info()