"""API route definitions"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
from pathlib import Path
from datetime import datetime

from config.settings import MODEL_INFO
from core.dependencies import get_current_model, validate_image_file
from core.model_loader import get_model_classes
from services.detection import run_inference, process_detections, generate_annotated_image
from services.processing import create_comprehensive_results, save_results

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