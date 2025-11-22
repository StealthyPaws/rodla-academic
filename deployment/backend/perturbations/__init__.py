"""
Perturbation Module for RoDLA API
==================================
Image perturbation methods for robustness testing.
"""

from .blur import apply_defocus, apply_vibration
from .content import add_watermark, add_image_background
from .noise import apply_speckle, apply_texture
from .inconsistency import apply_ink_holdout, apply_ink_bleeding, apply_illumination
from .spatial import apply_rotation, apply_keystoning, apply_warping
from .apply import apply_perturbation, apply_multiple_perturbations, PERTURBATION_CATEGORIES, get_perturbation_info

__all__ = [
    # Blur
    'apply_defocus',
    'apply_vibration',
    # Content
    'add_watermark',
    'add_image_background',
    # Noise
    'apply_speckle',
    'apply_texture',
    # Inconsistency
    'apply_ink_holdout',
    'apply_ink_bleeding',
    'apply_illumination',
    # Spatial
    'apply_rotation',
    'apply_keystoning',
    'apply_warping',
    # Main functions
    'apply_perturbation',
    'apply_multiple_perturbations',
    'get_perturbation_info',
    'PERTURBATION_CATEGORIES',
]