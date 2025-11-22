"""FastAPI dependency injection functions"""
from fastapi import HTTPException
from .model_loader import get_model


async def get_current_model():
    """
    Dependency to get the current loaded model
    
    Raises:
        HTTPException: If model is not loaded
        
    Returns:
        The loaded model instance
    """
    try:
        model = get_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please wait for initialization."
            )
        return model
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}"
        )


async def validate_image_file(content_type: str) -> bool:
    """
    Validate that uploaded file is an image
    
    Args:
        content_type: MIME type of uploaded file
        
    Returns:
        True if valid image
        
    Raises:
        HTTPException: If not a valid image
    """
    if not content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    return True