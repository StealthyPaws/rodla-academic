"""Pydantic models for API request/response validation"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class BBoxInfo(BaseModel):
    """Bounding box information"""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    center_x: float
    center_y: float


class Detection(BaseModel):
    """Single detection result"""
    class_id: int
    class_name: str
    bbox: BBoxInfo
    confidence: float
    area: float
    aspect_ratio: float


class ImageInfo(BaseModel):
    """Image metadata"""
    width: int
    height: int
    aspect_ratio: float
    total_pixels: int


class DetectionConfig(BaseModel):
    """Detection configuration"""
    score_threshold: float
    model: str
    framework: str
    max_detections: int


class DetectionRequest(BaseModel):
    """Request parameters for detection"""
    score_thr: float = Field(default=0.3, ge=0.0, le=1.0)
    return_image: bool = Field(default=False)
    save_json: bool = Field(default=True)
    generate_visualizations: bool = Field(default=True)


class CoreMetricsSummary(BaseModel):
    """Core detection metrics summary"""
    total_detections: int
    unique_classes: int
    average_confidence: float
    median_confidence: float
    min_confidence: float
    max_confidence: float
    coverage_percentage: float
    average_detection_area: float


class RoDLAMetrics(BaseModel):
    """RoDLA-specific metrics"""
    note: str
    estimated_mPE: float
    estimated_mRD: float
    confidence_std: float
    confidence_range: float
    robustness_score: float
    interpretation: Dict[str, str]


class DetectionResponse(BaseModel):
    """Complete detection response"""
    success: bool
    timestamp: str
    filename: str
    image_info: ImageInfo
    detection_config: DetectionConfig
    core_results: Dict[str, Any]
    rodla_metrics: RoDLAMetrics
    spatial_analysis: Dict[str, Any]
    class_analysis: Dict[str, Any]
    confidence_analysis: Dict[str, Any]
    robustness_indicators: Dict[str, Any]
    layout_complexity: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    interpretation: Dict[str, Any]
    all_detections: List[Detection]
    visualizations: Optional[Dict[str, str]] = None


# ============================================================================
# PERTURBATION SCHEMAS
# ============================================================================
from enum import Enum

class PerturbationType(str, Enum):
    """Available perturbation types."""
    # Blur
    defocus = "defocus"
    vibration = "vibration"
    # Noise
    speckle = "speckle"
    texture = "texture"
    # Content
    watermark = "watermark"
    background = "background"
    # Inconsistency
    ink_holdout = "ink_holdout"
    ink_bleeding = "ink_bleeding"
    illumination = "illumination"
    # Spatial
    rotation = "rotation"
    keystoning = "keystoning"
    warping = "warping"


class PerturbationConfig(BaseModel):
    """Single perturbation configuration."""
    type: PerturbationType = Field(
        ..., 
        description="Type of perturbation to apply"
    )
    degree: int = Field(
        1, 
        ge=1, 
        le=3, 
        description="Intensity: 1=mild, 2=moderate, 3=severe"
    )
    background_folder: Optional[str] = Field(
        None,
        description="Path to background images (only for 'background' type)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "type": "defocus",
                "degree": 2
            }
        }


class PerturbationResponse(BaseModel):
    """Response model for perturbation operations."""
    success: bool
    message: str
    perturbations_applied: List[Dict[str, Any]]
    perturbations_failed: List[Dict[str, Any]]
    total_perturbations: int
    success_rate: float
    image_base64: Optional[str] = None
    saved_path: Optional[str] = None
    original_shape: List[int]
    perturbed_shape: List[int]


class PerturbationInfoResponse(BaseModel):
    """Response model for perturbation information endpoint."""
    total_perturbations: int
    categories: Dict[str, Dict[str, Any]]
    all_types: List[str]
    degree_levels: Dict[int, str]
    notes: Dict[str, str]