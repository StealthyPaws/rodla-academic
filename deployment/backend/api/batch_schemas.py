"""
Pydantic Models for Batch Processing API
=========================================
Request and response schemas for batch operations.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class BatchJobConfig(BaseModel):
    """Configuration for a batch processing job."""
    total_images: int = Field(..., ge=1, le=300, description="Total number of images")
    score_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Confidence threshold")
    visualization_mode: str = Field(
        "none", 
        pattern="^(none|per_image|summary|both)$",
        description="Visualization generation mode"
    )
    perturbations: Optional[List] = Field(
        None, 
        description="Perturbation configurations (shared or per-image)"
    )
    perturbation_mode: str = Field(
        "shared",
        pattern="^(shared|per_image)$",
        description="Perturbation application mode"
    )
    save_json: bool = Field(True, description="Save results to disk")
    background_folder: Optional[str] = Field(None, description="Background images folder")
    filenames: List[str] = Field(..., description="Original filenames")
    
    class Config:
        schema_extra = {
            "example": {
                "total_images": 10,
                "score_threshold": 0.3,
                "visualization_mode": "summary",
                "perturbations": [{"type": "defocus", "degree": 2}],
                "perturbation_mode": "shared",
                "save_json": True,
                "filenames": ["img1.jpg", "img2.jpg", "img3.jpg"]
            }
        }


class BatchJobProgress(BaseModel):
    """Progress tracking for batch job."""
    current: int = Field(..., ge=0, description="Images processed so far")
    total: int = Field(..., ge=1, description="Total images to process")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Completion percentage")
    successful: int = Field(0, ge=0, description="Successfully processed images")
    failed: int = Field(0, ge=0, description="Failed images")
    processing_times: List[float] = Field(
        default_factory=list,
        description="Processing time for each image (seconds)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "current": 5,
                "total": 10,
                "percentage": 50.0,
                "successful": 4,
                "failed": 1,
                "processing_times": [2.3, 1.8, 3.1, 2.0, 2.5]
            }
        }


class BatchImageResult(BaseModel):
    """Result for a single image in batch."""
    image: str = Field(..., description="Image filename")
    status: str = Field(..., pattern="^(success|failed)$", description="Processing status")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    detections_count: Optional[int] = Field(None, description="Number of detections")
    average_confidence: Optional[float] = Field(None, description="Average confidence score")
    result_path: Optional[str] = Field(None, description="Path to result JSON file")
    error: Optional[str] = Field(None, description="Error message if failed")
    traceback: Optional[str] = Field(None, description="Full traceback if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "image": "document1.jpg",
                "status": "success",
                "processing_time": 2.3,
                "detections_count": 47,
                "average_confidence": 0.82,
                "result_path": "outputs/batch_20241123_143022/image_001/results.json"
            }
        }


class BatchJobSummary(BaseModel):
    """Aggregate statistics for completed batch."""
    total_images: int = Field(..., description="Total images in batch")
    successful_images: int = Field(..., description="Successfully processed images")
    failed_images: int = Field(..., description="Failed images")
    total_detections: int = Field(..., description="Total detections across all images")
    average_detections_per_image: float = Field(..., description="Average detections per image")
    average_confidence: float = Field(..., description="Average confidence across all detections")
    confidence_std: float = Field(..., description="Standard deviation of confidence")
    class_distribution: Dict[str, int] = Field(..., description="Total count per class")
    processing_time_total: float = Field(..., description="Total processing time (seconds)")
    processing_time_average: float = Field(..., description="Average time per image (seconds)")
    
    class Config:
        schema_extra = {
            "example": {
                "total_images": 10,
                "successful_images": 10,
                "failed_images": 0,
                "total_detections": 456,
                "average_detections_per_image": 45.6,
                "average_confidence": 0.79,
                "confidence_std": 0.12,
                "class_distribution": {
                    "paragraph": 152,
                    "title": 84,
                    "figure": 67
                },
                "processing_time_total": 23.5,
                "processing_time_average": 2.35
            }
        }


class BatchJobStatus(BaseModel):
    """Complete status of a batch job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(
        ...,
        pattern="^(queued|processing|completed|partial|failed)$",
        description="Current job status"
    )
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    started_at: Optional[str] = Field(None, description="Processing start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    config: BatchJobConfig = Field(..., description="Job configuration")
    progress: BatchJobProgress = Field(..., description="Processing progress")
    results: List[BatchImageResult] = Field(..., description="Per-image results")
    summary: Optional[BatchJobSummary] = Field(None, description="Aggregate statistics")
    output_dir: Optional[str] = Field(None, description="Batch output directory path")
    error: Optional[str] = Field(None, description="Error message if job failed")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "processing",
                "created_at": "2024-11-23T14:30:22.123456",
                "updated_at": "2024-11-23T14:32:45.789012",
                "started_at": "2024-11-23T14:30:23.456789",
                "completed_at": None,
                "config": {},
                "progress": {},
                "results": [],
                "summary": None,
                "output_dir": None,
                "error": None
            }
        }


class BatchJobCreateResponse(BaseModel):
    """Response when creating a new batch job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field("queued", description="Initial status (always 'queued')")
    message: str = Field(..., description="Human-readable message")
    total_images: int = Field(..., description="Number of images to process")
    status_endpoint: str = Field(..., description="Endpoint to poll for status")
    estimated_time_seconds: int = Field(..., description="Estimated processing time")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "queued",
                "message": "Batch processing started for 10 images",
                "total_images": 10,
                "status_endpoint": "/api/batch-job/3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "estimated_time_seconds": 30
            }
        }