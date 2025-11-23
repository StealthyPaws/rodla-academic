"""
Batch Processing Service
========================
Main orchestration logic for processing multiple images in background.
"""

import os
import cv2
import numpy as np
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import time
import json

from config.settings import (
    OUTPUT_DIR,
    BATCH_OUTPUT_PREFIX,
    ESTIMATED_TIME_PER_IMAGE
)
from services.batch_job_manager import BatchJobManager
from services.detection import run_inference, process_detections
from services.processing import create_comprehensive_results, save_results
from services.perturbation import perturb_image_service
from utils.serialization import convert_to_json_serializable


async def process_batch_job(
    job_id: str,
    temp_file_paths: List[str],
    config: dict,
    job_manager: BatchJobManager
):
    """
    Process a batch of images in the background.
    
    This is the main background task that:
    1. Processes each image sequentially
    2. Updates job progress after each image
    3. Handles individual image failures gracefully
    4. Generates summary statistics at the end
    
    Args:
        job_id: Unique job identifier
        temp_file_paths: List of temporary file paths for uploaded images
        config: Job configuration dictionary
        job_manager: BatchJobManager instance
    
    Returns:
        None (updates job status via job_manager)
    """
    try:
        print("="*60)
        print(f"🚀 Starting batch job: {job_id}")
        print("="*60)
        
        # Mark job as started
        job_manager.mark_job_started(job_id)
        
        # Create batch output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = OUTPUT_DIR / f"{BATCH_OUTPUT_PREFIX}{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Batch output directory: {batch_dir}")
        
        # Process each image sequentially
        total_images = len(temp_file_paths)
        
        for i, temp_path in enumerate(temp_file_paths):
            filename = config["filenames"][i]
            image_index = i + 1
            
            print(f"\n{'='*60}")
            print(f"📸 Processing image {image_index}/{total_images}: {filename}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # Process single image
                result = await process_single_image(
                    temp_path=temp_path,
                    filename=filename,
                    image_index=image_index,
                    batch_dir=batch_dir,
                    config=config,
                    job_id=job_id
                )
                
                processing_time = time.time() - start_time
                
                # Add successful result
                job_manager.add_job_result(job_id, filename, {
                    "status": "success",
                    "processing_time": round(processing_time, 2),
                    "detections_count": result["detections_count"],
                    "average_confidence": result["average_confidence"],
                    "result_path": result["result_path"]
                })
                
                # Update progress
                job_manager.update_job_progress(
                    job_id,
                    current=image_index,
                    total=total_images,
                    processing_time=round(processing_time, 2)
                )
                
                print(f"✅ Completed in {processing_time:.2f}s")
                
            except Exception as e:
                # Handle individual image failure
                processing_time = time.time() - start_time
                error_msg = str(e)
                error_trace = traceback.format_exc()
                
                print(f"❌ Failed: {error_msg}")
                
                # Add failed result
                job_manager.add_job_result(job_id, filename, {
                    "status": "failed",
                    "error": error_msg,
                    "traceback": error_trace
                })
                
                # Update progress (still count as processed)
                job_manager.update_job_progress(
                    job_id,
                    current=image_index,
                    total=total_images,
                    processing_time=round(processing_time, 2)
                )
            
            finally:
                # Cleanup temp file for this image
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to cleanup {temp_path}: {e}")
        
        # Generate summary if requested
        summary = None
        if config["visualization_mode"] in ["summary", "both"]:
            print(f"\n{'='*60}")
            print("📊 Generating batch summary...")
            print(f"{'='*60}")
            
            try:
                summary = generate_batch_summary(
                    job_id=job_id,
                    job_manager=job_manager,
                    batch_dir=batch_dir,
                    config=config
                )
                print("✅ Summary generated successfully")
            except Exception as e:
                print(f"⚠️  Warning: Failed to generate summary: {e}")
                traceback.print_exc()
        
        # Mark job as completed
        job_manager.mark_job_completed(
            job_id=job_id,
            summary=summary,
            output_dir=str(batch_dir)
        )
        
        print(f"\n{'='*60}")
        print(f"🎉 Batch job completed: {job_id}")
        print(f"{'='*60}")
        
    except Exception as e:
        # Critical error - mark entire job as failed
        print(f"\n{'='*60}")
        print(f"💥 Critical error in batch job {job_id}")
        print(f"{'='*60}")
        traceback.print_exc()
        
        job_manager.mark_job_failed(job_id, f"Critical error: {str(e)}")
        
        # Cleanup all remaining temp files
        for temp_path in temp_file_paths:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


async def process_single_image(
    temp_path: str,
    filename: str,
    image_index: int,
    batch_dir: Path,
    config: dict,
    job_id: str
) -> dict:
    """
    Process a single image within a batch.
    
    Args:
        temp_path: Path to temporary uploaded file
        filename: Original filename
        image_index: Index in batch (1-based)
        batch_dir: Batch output directory
        config: Job configuration
        job_id: Job identifier for logging
    
    Returns:
        Dictionary with:
            - detections_count: Number of detections
            - average_confidence: Average confidence score
            - result_path: Path to saved results JSON
    
    Raises:
        Exception: If processing fails
    """
    perturbed_temp_path = None
    
    try:
        # Read image
        image = cv2.imread(temp_path)
        if image is None:
            raise ValueError(f"Failed to read image: {temp_path}")
        
        print(f"   📏 Image shape: {image.shape}")
        
        # Apply perturbations if configured
        if config.get("perturbations"):
            print(f"   🎨 Applying perturbations...")
            
            perturbations = get_perturbations_for_image(config, image_index - 1)
            
            if perturbations:
                pert_result, image = perturb_image_service(
                    image=image,
                    perturbations=perturbations,
                    filename=filename,
                    save_image=False,  # Don't save perturbed images by default in batch
                    return_base64=False,
                    background_folder=config.get("background_folder")
                )
                
                applied_count = len(pert_result["perturbations_applied"])
                print(f"   ✓ Applied {applied_count} perturbations")
        
        # Save perturbed image to temp file for detection
        perturbed_temp_path = temp_path + "_perturbed.jpg"
        cv2.imwrite(perturbed_temp_path, image)
        
        # Run detection
        print(f"   🔍 Running detection...")
        result, img_width, img_height = run_inference(perturbed_temp_path)
        
        # Process detections
        detections = process_detections(result, config["score_threshold"])
        print(f"   ✓ Found {len(detections)} detections")
        
        # Generate visualizations based on mode
        generate_viz = config["visualization_mode"] in ["per_image", "both"]
        
        if generate_viz:
            print(f"   📊 Generating visualizations...")
        
        # Create comprehensive results
        results_dict = create_comprehensive_results(
            detections=detections,
            img_width=img_width,
            img_height=img_height,
            filename=filename,
            score_threshold=config["score_threshold"],
            generate_viz=generate_viz
        )
        
        # Create individual image directory
        image_dir = batch_dir / f"image_{image_index:03d}"
        image_dir.mkdir(exist_ok=True)
        
        # Save individual result
        if config.get("save_json", True):
            save_individual_result(results_dict, image_dir, filename)
            print(f"   💾 Saved results to {image_dir}")
        
        # Return summary info
        return {
            "detections_count": len(detections),
            "average_confidence": results_dict["core_results"]["summary"]["average_confidence"],
            "result_path": str(image_dir / "results.json")
        }
        
    finally:
        # Cleanup perturbed temp file
        if perturbed_temp_path and os.path.exists(perturbed_temp_path):
            try:
                os.unlink(perturbed_temp_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to cleanup {perturbed_temp_path}: {e}")


def get_perturbations_for_image(config: dict, image_index: int) -> List[dict]:
    """
    Get perturbations for a specific image based on mode.
    
    Args:
        config: Job configuration with perturbations
        image_index: Image index (0-based)
    
    Returns:
        List of perturbation dictionaries for this image
    """
    perturbations = config.get("perturbations")
    if not perturbations:
        return []
    
    perturbation_mode = config.get("perturbation_mode", "shared")
    
    if perturbation_mode == "per_image":
        # Per-image mode: perturbations is list of lists
        return perturbations[image_index]
    else:
        # Shared mode: same perturbations for all images
        return perturbations


def save_individual_result(
    results_dict: dict,
    image_dir: Path,
    filename: str
):
    """
    Save individual image results to directory.
    
    Structure:
        image_dir/
        ├── results.json           (Full results without visualizations)
        └── visualizations/        (If generated)
            ├── class_distribution.png
            ├── confidence_histogram.png
            └── ...
    
    Args:
        results_dict: Complete results dictionary
        image_dir: Directory to save results
        filename: Original filename
    """
    # Save JSON (without base64 visualizations)
    json_results = {k: v for k, v in results_dict.items() if k != "visualizations"}
    json_results = convert_to_json_serializable(json_results)
    
    json_path = image_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save visualizations if present
    if results_dict.get("visualizations"):
        viz_dir = image_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        import base64
        for viz_name, viz_base64 in results_dict["visualizations"].items():
            if viz_base64:
                viz_path = viz_dir / f"{viz_name}.png"
                
                # Decode base64 and save
                try:
                    img_data = base64.b64decode(viz_base64.split(',')[1])
                    with open(viz_path, 'wb') as f:
                        f.write(img_data)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to save {viz_name}: {e}")


def generate_batch_summary(
    job_id: str,
    job_manager: BatchJobManager,
    batch_dir: Path,
    config: dict
) -> dict:
    """
    Generate aggregate statistics and visualizations for entire batch.
    
    Args:
        job_id: Job identifier
        job_manager: Job manager instance
        batch_dir: Batch output directory
        config: Job configuration
    
    Returns:
        Summary statistics dictionary
    """
    job = job_manager.get_job(job_id)
    
    # Collect data from all successful results
    all_detections = []
    all_confidences = []
    class_counts = {}
    per_image_stats = []
    
    for result in job["results"]:
        if result["status"] == "success":
            # Load individual result file
            result_path = Path(result["result_path"])
            
            try:
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                
                # Collect detections
                detections = result_data.get("all_detections", [])
                all_detections.extend(detections)
                
                # Collect confidences
                for det in detections:
                    all_confidences.append(det["confidence"])
                
                # Aggregate class counts
                for class_name, class_data in result_data.get("class_analysis", {}).items():
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += class_data["count"]
                
                # Store per-image stats
                per_image_stats.append({
                    "image": result["image"],
                    "detections": result["detections_count"],
                    "avg_confidence": result["average_confidence"],
                    "processing_time": result["processing_time"],
                    "status": "success"
                })
                
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {result_path}: {e}")
    
    # Calculate summary statistics
    successful_count = job["progress"]["successful"]
    total_detections = len(all_detections)
    
    summary_stats = {
        "total_images": job["config"]["total_images"],
        "successful_images": successful_count,
        "failed_images": job["progress"]["failed"],
        "total_detections": total_detections,
        "average_detections_per_image": round(
            total_detections / max(successful_count, 1), 2
        ),
        "average_confidence": round(np.mean(all_confidences), 4) if all_confidences else 0,
        "confidence_std": round(np.std(all_confidences), 4) if all_confidences else 0,
        "confidence_min": round(np.min(all_confidences), 4) if all_confidences else 0,
        "confidence_max": round(np.max(all_confidences), 4) if all_confidences else 0,
        "class_distribution": class_counts,
        "processing_time_total": round(sum(job["progress"]["processing_times"]), 2),
        "processing_time_average": round(
            np.mean(job["progress"]["processing_times"]), 2
        ) if job["progress"]["processing_times"] else 0,
        "processing_time_min": round(
            np.min(job["progress"]["processing_times"]), 2
        ) if job["progress"]["processing_times"] else 0,
        "processing_time_max": round(
            np.max(job["progress"]["processing_times"]), 2
        ) if job["progress"]["processing_times"] else 0
    }
    
    # Save summary to file
    summary_path = batch_dir / "summary.json"
    summary_output = {
        "batch_id": batch_dir.name,
        "job_id": job_id,
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
        "statistics": summary_stats,
        "per_image_summary": per_image_stats
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_output, f, indent=2)
    
    print(f"   💾 Saved summary to {summary_path}")
    
    # Generate summary visualizations if in summary or both mode
    if config["visualization_mode"] in ["summary", "both"]:
        print(f"   📊 Generating summary visualizations...")
        
        try:
            from services.visualization import generate_summary_visualizations
            
            summary_viz = generate_summary_visualizations(
                summary_stats=summary_stats,
                all_detections=all_detections,
                processing_times=job["progress"]["processing_times"],
                per_image_stats=per_image_stats
            )
            
            # Save summary visualizations
            viz_dir = batch_dir / "summary_visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            import base64
            for viz_name, viz_base64 in summary_viz.items():
                if viz_base64:
                    viz_path = viz_dir / f"{viz_name}.png"
                    
                    try:
                        img_data = base64.b64decode(viz_base64.split(',')[1])
                        with open(viz_path, 'wb') as f:
                            f.write(img_data)
                        print(f"   ✓ Saved {viz_name}.png")
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to save {viz_name}: {e}")
            
            summary_output["visualizations_generated"] = list(summary_viz.keys())
            
            # Re-save summary with visualization info
            with open(summary_path, 'w') as f:
                json.dump(summary_output, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to generate summary visualizations: {e}")
            traceback.print_exc()
    
    return summary_stats


def save_uploaded_file_temp(file) -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        file: FastAPI UploadFile object
    
    Returns:
        Path to temporary file
    """
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        return tmp.name