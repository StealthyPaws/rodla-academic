"""
Perturbation Application Module - Using Common Libraries
Applies 12 document degradation perturbations using PIL, OpenCV, NumPy, and SciPy
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from typing import Optional, Tuple, List, Dict
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import random


def encode_to_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is in RGB format"""
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


# ============================================================================
# BLUR PERTURBATIONS
# ============================================================================

def apply_defocus(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply defocus blur (Gaussian blur simulating out-of-focus camera)
    degree: 1 (mild), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No defocus"
    
    try:
        image = encode_to_rgb(image)
        
        # Kernel sizes for different degrees
        kernel_sizes = {1: 3, 2: 7, 3: 15}
        kernel_size = kernel_sizes.get(degree, 15)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return blurred, True, f"Defocus applied (kernel={kernel_size})"
    except Exception as e:
        return image, False, f"Defocus error: {str(e)}"


def apply_vibration(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply motion blur (vibration/camera shake effect)
    degree: 1 (mild), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No vibration"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Motion blur kernel sizes
        kernel_sizes = {1: 5, 2: 15, 3: 25}
        kernel_size = kernel_sizes.get(degree, 25)
        
        # Create motion blur kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred, True, f"Vibration applied (kernel={kernel_size})"
    except Exception as e:
        return image, False, f"Vibration error: {str(e)}"


# ============================================================================
# NOISE PERTURBATIONS
# ============================================================================

def apply_speckle(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply speckle noise (multiplicative noise)
    degree: 1 (mild), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No speckle"
    
    try:
        image = encode_to_rgb(image)
        image_float = image.astype(np.float32) / 255.0
        
        # Noise intensity
        noise_levels = {1: 0.1, 2: 0.25, 3: 0.5}
        noise_level = noise_levels.get(degree, 0.5)
        
        # Generate speckle noise
        speckle = np.random.normal(1, noise_level, image_float.shape)
        noisy = image_float * speckle
        
        # Clip values
        noisy = np.clip(noisy, 0, 1)
        noisy = (noisy * 255).astype(np.uint8)
        
        return noisy, True, f"Speckle applied (intensity={noise_level})"
    except Exception as e:
        return image, False, f"Speckle error: {str(e)}"


def apply_texture(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply texture/grain noise (additive Gaussian noise)
    degree: 1 (mild), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No texture"
    
    try:
        image = encode_to_rgb(image)
        image_float = image.astype(np.float32)
        
        # Noise levels
        noise_levels = {1: 10, 2: 25, 3: 50}
        noise_level = noise_levels.get(degree, 50)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, image_float.shape)
        noisy = image_float + noise
        
        # Clip values
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy, True, f"Texture applied (std={noise_level})"
    except Exception as e:
        return image, False, f"Texture error: {str(e)}"


# ============================================================================
# CONTENT PERTURBATIONS
# ============================================================================

def apply_watermark(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Add watermark text overlay
    degree: 1 (subtle), 2 (noticeable), 3 (heavy)
    """
    if degree == 0:
        return image, True, "No watermark"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Convert to PIL for text drawing
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image, 'RGBA')
        
        # Watermark parameters by degree
        watermark_text = "WATERMARK" * degree
        fontsize_list = {1: max(10, h // 20), 2: max(15, h // 15), 3: max(20, h // 10)}
        fontsize = fontsize_list.get(degree, 20)
        
        alpha_list = {1: 64, 2: 128, 3: 200}
        alpha = alpha_list.get(degree, 200)
        
        # Draw watermark multiple times
        num_watermarks = {1: 1, 2: 3, 3: 5}.get(degree, 5)
        
        for i in range(num_watermarks):
            x = (w // (num_watermarks + 1)) * (i + 1)
            y = h // 2
            color = (255, 0, 0, alpha)
            draw.text((x, y), watermark_text, fill=color)
        
        return np.array(pil_image), True, f"Watermark applied (degree={degree})"
    except Exception as e:
        return image, False, f"Watermark error: {str(e)}"


def apply_background(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Add background patterns/textures
    degree: 1 (subtle), 2 (noticeable), 3 (heavy)
    """
    if degree == 0:
        return image, True, "No background"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Create background pattern
        pattern_intensity = {1: 0.1, 2: 0.2, 3: 0.35}.get(degree, 0.35)
        
        # Generate random pattern
        pattern = np.random.randint(0, 100, (h, w, 3), dtype=np.uint8)
        pattern = cv2.GaussianBlur(pattern, (21, 21), 0)
        
        # Blend with original image
        result = cv2.addWeighted(image, 1.0, pattern, pattern_intensity, 0)
        
        return result.astype(np.uint8), True, f"Background applied (intensity={pattern_intensity})"
    except Exception as e:
        return image, False, f"Background error: {str(e)}"


# ============================================================================
# INCONSISTENCY PERTURBATIONS
# ============================================================================

def apply_ink_holdout(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply ink holdout (missing ink/text drop-out)
    degree: 1 (few gaps), 2 (some gaps), 3 (many gaps)
    """
    if degree == 0:
        return image, True, "No ink holdout"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Create white mask to simulate missing ink
        num_dropouts = {1: 3, 2: 8, 3: 15}.get(degree, 15)
        
        result = image.copy()
        
        for _ in range(num_dropouts):
            # Random position and size
            x = np.random.randint(0, w - 20)
            y = np.random.randint(0, h - 20)
            size = np.random.randint(10, 40)
            
            # Create white rectangle (simulating ink dropout)
            result[y:y+size, x:x+size] = [255, 255, 255]
        
        return result, True, f"Ink holdout applied (dropouts={num_dropouts})"
    except Exception as e:
        return image, False, f"Ink holdout error: {str(e)}"


def apply_ink_bleeding(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply ink bleeding effect (ink spread/bleed)
    degree: 1 (mild), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No ink bleeding"
    
    try:
        image = encode_to_rgb(image)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Dilate dark regions (simulating ink spread)
        kernel_sizes = {1: 3, 2: 5, 3: 7}
        kernel_size = kernel_sizes.get(degree, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate to spread ink
        dilated = cv2.dilate(gray, kernel, iterations=degree)
        
        # Blend back with original
        result = image.copy().astype(np.float32)
        result[:,:,0] = cv2.addWeighted(image[:,:,0], 0.7, dilated, 0.3, 0)
        result[:,:,1] = cv2.addWeighted(image[:,:,1], 0.7, dilated, 0.3, 0)
        result[:,:,2] = cv2.addWeighted(image[:,:,2], 0.7, dilated, 0.3, 0)
        
        return np.clip(result, 0, 255).astype(np.uint8), True, f"Ink bleeding applied (degree={degree})"
    except Exception as e:
        return image, False, f"Ink bleeding error: {str(e)}"


def apply_illumination(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply illumination variations (uneven lighting)
    degree: 1 (subtle), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No illumination"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Create illumination pattern
        intensity = {1: 0.15, 2: 0.3, 3: 0.5}.get(degree, 0.5)
        
        # Create gradient-like illumination from corners
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create vignette effect
        illumination = 1 - intensity * (np.sqrt(X**2 + Y**2) / np.sqrt(2))
        illumination = np.clip(illumination, 0, 1)
        
        # Apply to each channel
        result = image.astype(np.float32)
        for c in range(3):
            result[:,:,c] = result[:,:,c] * illumination
        
        return np.clip(result, 0, 255).astype(np.uint8), True, f"Illumination applied (intensity={intensity})"
    except Exception as e:
        return image, False, f"Illumination error: {str(e)}"


# ============================================================================
# SPATIAL PERTURBATIONS
# ============================================================================

def apply_rotation(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply rotation
    degree: 1 (±5°), 2 (±10°), 3 (±15°)
    """
    if degree == 0:
        return image, True, "No rotation"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Angle ranges by degree
        angle_ranges = {1: 5, 2: 10, 3: 15}
        max_angle = angle_ranges.get(degree, 15)
        
        # Random angle
        angle = np.random.uniform(-max_angle, max_angle)
        
        # Rotation matrix
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation with white padding
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderValue=(255, 255, 255))
        
        return rotated, True, f"Rotation applied (angle={angle:.1f}°)"
    except Exception as e:
        return image, False, f"Rotation error: {str(e)}"


def apply_keystoning(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply keystoning effect (perspective distortion)
    degree: 1 (subtle), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No keystoning"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Distortion amount
        distortion = {1: w * 0.05, 2: w * 0.1, 3: w * 0.15}.get(degree, w * 0.15)
        
        # Source corners
        src_points = np.float32([
            [0, 0],
            [w - 1, 0],
            [0, h - 1],
            [w - 1, h - 1]
        ])
        
        # Destination corners (with perspective distortion)
        dst_points = np.float32([
            [distortion, 0],
            [w - 1 - distortion * 0.5, 0],
            [0, h - 1],
            [w - 1, h - 1]
        ])
        
        # Get perspective transform
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (w, h), borderValue=(255, 255, 255))
        
        return warped, True, f"Keystoning applied (distortion={distortion:.1f})"
    except Exception as e:
        return image, False, f"Keystoning error: {str(e)}"


def apply_warping(image: np.ndarray, degree: int) -> Tuple[np.ndarray, bool, str]:
    """
    Apply elastic/elastic deformation
    degree: 1 (mild), 2 (moderate), 3 (severe)
    """
    if degree == 0:
        return image, True, "No warping"
    
    try:
        image = encode_to_rgb(image)
        h, w = image.shape[:2]
        
        # Warping parameters
        alpha_values = {1: 15, 2: 30, 3: 60}
        sigma_values = {1: 3, 2: 5, 3: 8}
        alpha = alpha_values.get(degree, 60)
        sigma = sigma_values.get(degree, 8)
        
        # Generate random displacement field
        dx = np.random.randn(h, w) * sigma
        dy = np.random.randn(h, w) * sigma
        
        # Smooth displacement field
        dx = gaussian_filter(dx, sigma=sigma) * alpha
        dy = gaussian_filter(dy, sigma=sigma) * alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        x_warped = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_warped = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Remap image
        warped = cv2.remap(image, x_warped, y_warped, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        
        return warped, True, f"Warping applied (alpha={alpha}, sigma={sigma})"
    except Exception as e:
        return image, False, f"Warping error: {str(e)}"


# ============================================================================
# Main Perturbation Application
# ============================================================================

PERTURBATION_FUNCTIONS = {
    # Blur
    "defocus": apply_defocus,
    "vibration": apply_vibration,
    # Noise
    "speckle": apply_speckle,
    "texture": apply_texture,
    # Content
    "watermark": apply_watermark,
    "background": apply_background,
    # Inconsistency
    "ink_holdout": apply_ink_holdout,
    "ink_bleeding": apply_ink_bleeding,
    "illumination": apply_illumination,
    # Spatial
    "rotation": apply_rotation,
    "keystoning": apply_keystoning,
    "warping": apply_warping,
}


def apply_perturbation(
    image: np.ndarray,
    perturbation_type: str,
    degree: int = 1
) -> Tuple[np.ndarray, bool, str]:
    """
    Apply a single perturbation to an image
    
    Args:
        image: Input image as numpy array (BGR or RGB)
        perturbation_type: Type of perturbation (see PERTURBATION_FUNCTIONS)
        degree: Severity level (1=mild, 2=moderate, 3=severe)
    
    Returns:
        Tuple of (result_image, success, message)
    """
    if perturbation_type not in PERTURBATION_FUNCTIONS:
        return image, False, f"Unknown perturbation type: {perturbation_type}"
    
    if degree < 0 or degree > 3:
        return image, False, f"Invalid degree: {degree} (must be 0-3)"
    
    func = PERTURBATION_FUNCTIONS[perturbation_type]
    return func(image, degree)


def apply_multiple_perturbations(
    image: np.ndarray,
    perturbations: List[Tuple[str, int]]
) -> Tuple[np.ndarray, bool, str]:
    """
    Apply multiple perturbations in sequence
    
    Args:
        image: Input image
        perturbations: List of (type, degree) tuples
    
    Returns:
        Tuple of (result_image, success, message)
    """
    result = image.copy()
    messages = []
    
    for ptype, degree in perturbations:
        result, success, msg = apply_perturbation(result, ptype, degree)
        messages.append(msg)
        if not success:
            return image, False, f"Failed: {msg}"
    
    return result, True, " | ".join(messages)


def get_perturbation_info() -> Dict:
    """Get information about all available perturbations"""
    return {
        "total_perturbations": len(PERTURBATION_FUNCTIONS),
        "types": list(PERTURBATION_FUNCTIONS.keys()),
        "categories": {
            "blur": ["defocus", "vibration"],
            "noise": ["speckle", "texture"],
            "content": ["watermark", "background"],
            "inconsistency": ["ink_holdout", "ink_bleeding", "illumination"],
            "spatial": ["rotation", "keystoning", "warping"]
        }
    }
