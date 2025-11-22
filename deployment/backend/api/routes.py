"""API route definitions"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
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