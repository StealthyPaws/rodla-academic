"""API route definitions"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends,BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
import tempfile
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List
import cv2

from config.settings import MODEL_INFO
from core.dependencies import get_current_model, validate_image_file
from core.dependencies import get_model
from core.model_loader import get_model_classes
from services.detection import run_inference, process_detections, generate_annotated_image
from services.processing import create_comprehensive_results, save_results
from services.perturbation import (
    perturb_image_service,
    get_perturbation_info_service
)
from api.schemas import PerturbationConfig, PerturbationResponse
import asyncio



router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns:
        JSON with service health status
    """
    try:
        model = await get_current_model()
        return JSONResponse({
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        })
    except HTTPException:
        return JSONResponse(
            {
                "status": "unhealthy",
                "model_loaded": False,
                "error": "Model not loaded",
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )


@router.get("/api/model-info")
async def get_model_info(model=Depends(get_current_model)):
    """
    Get model information and capabilities
    
    Returns:
        JSON with model information
    """
    model_info = MODEL_INFO.copy()
    model_info["num_classes"] = len(get_model_classes())
    model_info["classes"] = get_model_classes()
    
    return JSONResponse(model_info)


@router.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    score_thr: str = Form("0.3"),
    return_image: str = Form("false"),
    save_json: str = Form("true"),
    generate_visualizations: str = Form("true"),
    model=Depends(get_current_model)
):
    """
    Comprehensive document layout detection with RoDLA metrics
    
    Args:
        file: Image file to process
        score_thr: Confidence threshold (0-1)
        return_image: Return annotated image instead of JSON
        save_json: Save results to JSON file
        generate_visualizations: Generate metric visualizations
        model: Injected model dependency
        
    Returns:
        JSONResponse with detection results or FileResponse with annotated image
    """
    tmp_path = None
    
    try:
        # Validate file type
        await validate_image_file(file.content_type)
        
        # Parse parameters
        score_threshold = float(score_thr)
        if not 0 <= score_threshold <= 1:
            raise HTTPException(400, "score_thr must be between 0 and 1")
            
        should_return_image = return_image.lower() in ('true', '1', 'yes')
        should_save_json = save_json.lower() in ('true', '1', 'yes')
        should_generate_viz = generate_visualizations.lower() in ('true', '1', 'yes')
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Run inference
        result, img_width, img_height = run_inference(tmp_path)
        
        # Process detections
        detections = process_detections(result, score_threshold)
        
        # Create comprehensive results
        results = create_comprehensive_results(
            detections=detections,
            img_width=img_width,
            img_height=img_height,
            filename=file.filename,
            score_threshold=score_threshold,
            generate_viz=should_generate_viz
        )
        
        # Save results if requested
        if should_save_json or not should_return_image:
            save_results(results, file.filename, should_generate_viz)
        
        # Return annotated image if requested
        if should_return_image:
            output_path = generate_annotated_image(
                tmp_path, result, score_threshold, file.filename
            )
            
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return FileResponse(
                path=str(output_path),
                media_type="image/jpeg",
                filename=f"annotated_{file.filename}"
            )
        
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return JSONResponse(results)
        
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except ValueError as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(400, f"Invalid parameter: {str(e)}")
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )
    
# ============================================================================
# PERTURBATION ENDPOINTS
# ============================================================================

@router.get("/api/perturbations/info")
async def get_perturbation_info():
    """
    Get information about available perturbations.
    
    Returns comprehensive details about:
    - All available perturbation types
    - Perturbation categories
    - Degree levels and their effects
    - Usage notes and requirements
    """
    try:
        info = get_perturbation_info_service()
        return JSONResponse(content=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get perturbation info: {str(e)}")


@router.post("/api/perturb")
async def perturb_image(
    file: UploadFile = File(..., description="Image file to perturb"),
    perturbations: str = Form(..., description="JSON array of perturbation configs"),
    return_base64: str = Form("false", description="Return perturbed image as base64"),
    save_image: str = Form("false", description="Save perturbed image to disk"),
    background_folder: Optional[str] = Form(None, description="Path to background images folder"),
):
    """
    Apply perturbations to an image without performing detection.
    
    **Parameters:**
    - **file**: Image file (JPEG, PNG, etc.)
    - **perturbations**: JSON string array of perturbation configurations
    - **return_base64**: "true" to return image as base64, "false" to return JSON only
    - **save_image**: "true" to save perturbed image to disk
    - **background_folder**: Optional path to background images (required for 'background' perturbation)
    
    **Example perturbations JSON:**
```json
    [
        {"type": "defocus", "degree": 2},
        {"type": "speckle", "degree": 1},
        {"type": "rotation", "degree": 1}
    ]
```
    
    **Returns:**
    - JSON with perturbation results and optionally base64 image
    - Or PNG image directly if return_base64="true" and only_image="true"
    """
    tmp_path = None
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Parse perturbations
        try:
            pert_configs = json.loads(perturbations)
            if not isinstance(pert_configs, list) or len(pert_configs) == 0:
                raise ValueError("Perturbations must be a non-empty array")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in perturbations parameter")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Read image
        image = cv2.imread(tmp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image file")
        
        # Convert form parameters to boolean
        return_base64_bool = return_base64.lower() == "true"
        save_image_bool = save_image.lower() == "true"
        
        # Apply perturbations
        result, perturbed_image = perturb_image_service(
            image=image,
            perturbations=pert_configs,
            filename=file.filename,
            save_image=save_image_bool,
            return_base64=return_base64_bool,
            background_folder=background_folder
        )
        
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        # Return response
        if return_base64_bool and result.get("image_base64"):
            # Return just the image if requested
            return JSONResponse(content=result)
        else:
            return JSONResponse(content=result)
    
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@router.post("/api/detect-with-perturbation")
async def detect_with_perturbation(
    file: UploadFile = File(..., description="Image file"),
    perturbations: Optional[str] = Form(None, description="JSON array of perturbations to apply before detection"),
    score_thr: str = Form("0.3", description="Confidence threshold"),
    return_image: str = Form("false", description="Return annotated image"),
    save_json: str = Form("true", description="Save results to JSON"),
    generate_visualizations: str = Form("true", description="Generate visualization charts"),
    save_perturbed_image: str = Form("false", description="Save the perturbed image"),
    background_folder: Optional[str] = Form(None, description="Background folder for 'background' perturbation"),
    model=Depends(get_model)
):
    """
    Apply perturbations to an image, then perform detection on the perturbed image.

    **Parameters:**
    - perturbations: JSON array of perturbation configs to apply first
    - save_perturbed_image: "true" to save perturbed image before detection
    - background_folder: Path to background images folder
    """
    tmp_path = None
    perturbed_tmp_path = None

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Read image
        image = cv2.imread(tmp_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image")

        # Apply perturbations if provided
        perturbation_results = None
        if perturbations:
            try:
                pert_configs = json.loads(perturbations)
                if isinstance(pert_configs, list) and len(pert_configs) > 0:
                    pert_result, image = perturb_image_service(
                        image=image,
                        perturbations=pert_configs,
                        filename=file.filename,
                        save_image=save_perturbed_image.lower() == "true",
                        return_base64=False,
                        background_folder=background_folder
                    )
                    perturbation_results = {
                        "applied": pert_result.get("perturbations_applied", []),
                        "failed": pert_result.get("perturbations_failed", []),
                        "success_rate": pert_result.get("success_rate", 0)
                    }
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in perturbations")

        # Save perturbed image temporarily for detection
        perturbed_tmp_path = tmp_path + "_perturbed.jpg"
        cv2.imwrite(perturbed_tmp_path, image)

        # Perform detection on perturbed image
        result, img_width, img_height = run_inference(perturbed_tmp_path)
        detections = process_detections(result, float(score_thr))

        # Create results JSON
        results_json = create_comprehensive_results(
            detections=detections,
            img_width=img_width,
            img_height=img_height,
            filename=file.filename,
            score_threshold=float(score_thr),
            generate_viz=generate_visualizations.lower() == "true"
        )

        # Save results if requested
        if save_json.lower() == "true":
            save_results(results_json, file.filename, generate_visualizations.lower() == "true")

        # Attach perturbation info
        if perturbation_results:
            results_json["perturbation_info"] = perturbation_results

        # Return results
        if return_image.lower() == "true":
            annotated_path = generate_annotated_image(perturbed_tmp_path, result, float(score_thr), file.filename)
            return FileResponse(
                path=str(annotated_path),
                media_type="image/jpeg",
                filename=f"annotated_{file.filename}"
            )
        else:
            return JSONResponse(content=results_json)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )
    finally:
        # Cleanup temporary files
        for path in [tmp_path, perturbed_tmp_path]:
            if path and os.path.exists(path):
                os.unlink(path)


# ============================================================================
# BATCH PROCESSING ENDPOINTS
# ============================================================================

@router.post("/api/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(..., description="1-300 image files to process"),
    score_thr: str = Form("0.3", description="Confidence threshold (0.0-1.0)"),
    perturbations: Optional[str] = Form(None, description="JSON array of perturbation configs"),
    visualization_mode: str = Form("none", description="Visualization mode: none|per_image|summary|both"),
    save_json: str = Form("true", description="Save results to disk"),
    background_folder: Optional[str] = Form(None, description="Background images folder path"),
    background_tasks: BackgroundTasks = None
):
    temp_file_paths = []

    try:
        from config.settings import (
            MAX_BATCH_SIZE,
            MIN_BATCH_SIZE,
            VISUALIZATION_MODES,
            ESTIMATED_TIME_PER_IMAGE,
            MAX_PERTURBATIONS_PER_REQUEST,
        )
        from services.batch_job_manager import get_job_manager
        from services.batch_processing import process_batch_job, save_uploaded_file_temp

        print("\n" + "=" * 60)
        print("📥 Batch Detection Request Received")
        print("=" * 60)

        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================

        # 1. Validate file count
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="At least one file is required")

        if len(files) < MIN_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Minimum {MIN_BATCH_SIZE} file(s) required, got {len(files)}",
            )

        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_BATCH_SIZE} files allowed, got {len(files)}",
            )

        print(f"✓ File count valid: {len(files)} images")

        # 2. Validate file types
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i+1} ({file.filename}) is not an image. Content-Type: {file.content_type}",
                )

        print(f"✓ All files are valid images")

        # 3. Parse and validate score threshold
        try:
            score_threshold = float(score_thr)
            if not 0 <= score_threshold <= 1:
                raise ValueError("Must be between 0 and 1")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid score_thr: {e}")

        print(f"✓ Score threshold: {score_threshold}")

        # 4. Validate visualization mode
        if visualization_mode not in VISUALIZATION_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid visualization_mode '{visualization_mode}'. Must be one of: {VISUALIZATION_MODES}",
            )

        print(f"✓ Visualization mode: {visualization_mode}")

        # 5. Parse and validate perturbations
        perturbation_config = None
        perturbation_mode = "shared"

        if perturbations:
            try:
                pert_data = json.loads(perturbations)

                if not isinstance(pert_data, list):
                    raise ValueError("Perturbations must be a JSON array")

                if len(pert_data) == 0:
                    raise ValueError("Perturbations array cannot be empty")

                # Detect per-image mode
                if isinstance(pert_data[0], list):
                    perturbation_mode = "per_image"
                    print("   Detected per-image perturbation mode")

                    if len(pert_data) != len(files):
                        raise ValueError(
                            f"Per-image perturbations count ({len(pert_data)}) must match file count ({len(files)})"
                        )

                    for i, pert_list in enumerate(pert_data):
                        if not isinstance(pert_list, list):
                            raise ValueError(
                                f"Perturbations[{i}] must be an array, got {type(pert_list).__name__}"
                            )

                        if len(pert_list) > MAX_PERTURBATIONS_PER_REQUEST:
                            raise ValueError(
                                f"Image {i+1}: Maximum {MAX_PERTURBATIONS_PER_REQUEST} perturbations allowed, got {len(pert_list)}"
                            )

                        for j, pert in enumerate(pert_list):
                            validate_perturbation_config(pert, f"Image {i+1}, Perturbation {j+1}")

                    print(f"   ✓ Validated {len(pert_data)} per-image sets")

                else:
                    perturbation_mode = "shared"
                    print("   Detected shared perturbation mode")

                    if len(pert_data) > MAX_PERTURBATIONS_PER_REQUEST:
                        raise ValueError(
                            f"Maximum {MAX_PERTURBATIONS_PER_REQUEST} perturbations allowed, got {len(pert_data)}"
                        )

                    for i, pert in enumerate(pert_data):
                        validate_perturbation_config(pert, f"Perturbation {i+1}")

                    print(f"   ✓ Validated {len(pert_data)} shared perturbations")

                perturbation_config = pert_data

            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in perturbations parameter: {e}")

            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid perturbations: {e}")

        else:
            print("✓ No perturbations specified")

        # 6. Parse save_json
        should_save_json = save_json.lower() in ("true", "1", "yes")

        # ====================================================================
        # JOB CREATION PHASE
        # ====================================================================

        print("\n📋 Creating batch job...")

        job_config = {
            "total_images": len(files),
            "score_threshold": score_threshold,
            "visualization_mode": visualization_mode,
            "perturbations": perturbation_config,
            "perturbation_mode": perturbation_mode,
            "save_json": should_save_json,
            "background_folder": background_folder,
            "filenames": [f.filename for f in files],
        }

        job_manager = get_job_manager()
        job_id = job_manager.create_job(job_config)

        print(f"✓ Job created: {job_id}")

        # ====================================================================
        # FILE UPLOAD PHASE
        # ====================================================================

        print("\n💾 Saving uploaded files...")

        for i, file in enumerate(files):
            try:
                temp_path = save_uploaded_file_temp(file)
                temp_file_paths.append(temp_path)

                if (i + 1) % 10 == 0 or i + 1 == len(files):
                    print(f"   Saved {i + 1}/{len(files)}")

            except Exception as e:
                for path in temp_file_paths:
                    try:
                        if os.path.exists(path):
                            os.unlink(path)
                    except:
                        pass

                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save file {i+1} ({file.filename}): {str(e)}",
                )

        print("✓ All files saved")

        # ====================================================================
        # BACKGROUND TASK LAUNCH
        # ====================================================================

        print("\n🚀 Launching background task...")

        background_tasks.add_task(
            process_batch_job,
            job_id=job_id,
            temp_file_paths=temp_file_paths,
            config=job_config,
            job_manager=job_manager,
        )

        print("✓ Background task launched")

        # ====================================================================
        # RESPONSE
        # ====================================================================

        base_time = len(files) * ESTIMATED_TIME_PER_IMAGE

        if visualization_mode == "per_image":
            from config.settings import ESTIMATED_TIME_PER_VIZ
            base_time += len(files) * ESTIMATED_TIME_PER_VIZ * 8

        elif visualization_mode == "both":
            from config.settings import ESTIMATED_TIME_PER_VIZ
            base_time += len(files) * ESTIMATED_TIME_PER_VIZ * 8 + ESTIMATED_TIME_PER_VIZ * 8

        elif visualization_mode == "summary":
            from config.settings import ESTIMATED_TIME_PER_VIZ
            base_time += ESTIMATED_TIME_PER_VIZ * 8

        estimated_time = int(base_time)

        response_data = {
            "job_id": job_id,
            "status": "queued",
            "message": f"Batch processing started for {len(files)} images",
            "total_images": len(files),
            "status_endpoint": f"/api/batch-job/{job_id}",
            "estimated_time_seconds": estimated_time,
        }

        print("\n✅ Request accepted")
        print("=" * 60)

        return JSONResponse(content=response_data, status_code=202)

    except HTTPException:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass
        raise

    except Exception as e:
        for path in temp_file_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass

        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()

        return JSONResponse(
            content={"success": False, "error": f"Failed to create batch job: {str(e)}"},
            status_code=500,
        )


# ============================================================================
# GET JOB STATUS ENDPOINT
# ============================================================================

@router.get("/api/batch-job/{job_id}")
async def get_batch_job_status(job_id: str):
    try:
        from services.batch_job_manager import get_job_manager

        job_manager = get_job_manager()
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return JSONResponse(content=job)

    except HTTPException:
        raise

    except Exception as e:
        print(f"ERROR in get_batch_job_status: {e}")
        import traceback

        traceback.print_exc()

        return JSONResponse(
            content={"success": False, "error": f"Failed to retrieve job status: {str(e)}"},
            status_code=500,
        )


# ============================================================================
# HELPER: PERTURBATION VALIDATOR
# ============================================================================

def validate_perturbation_config(pert: dict, context: str = ""):
    from config.settings import PERTURBATION_CATEGORIES

    all_pert_types = [p for category in PERTURBATION_CATEGORIES.values() for p in category["types"]]

    if not isinstance(pert, dict):
        raise ValueError(f"{context}: Perturbation must be an object, got {type(pert).__name__}")

    if "type" not in pert:
        raise ValueError(f"{context}: Missing required field 'type'")

    if "degree" not in pert:
        raise ValueError(f"{context}: Missing required field 'degree'")

    pert_type = pert["type"]
    if pert_type not in all_pert_types:
        raise ValueError(
            f"{context}: Invalid perturbation type '{pert_type}'. Must be one of: {all_pert_types}"
        )

    try:
        degree = int(pert["degree"])
        if degree not in (1, 2, 3):
            raise ValueError("Must be 1, 2, or 3")
    except Exception as e:
        raise ValueError(f"{context}: Invalid degree value '{pert['degree']}'. {e}")

    # background perturbation is allowed without folder; job-level folder may be passed
    return