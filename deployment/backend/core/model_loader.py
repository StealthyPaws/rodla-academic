"""Model loading and initialization"""
from typing import Optional
import torch
import gc
from mmdet.apis import init_detector
from config.settings import MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH

_model_instance: Optional[object] = None


def get_device() -> str:
    """Get the best available device"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load the RoDLA model with proper memory management"""
    global _model_instance
    
    if _model_instance is not None:
        return _model_instance
    
    try:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("🔄 Loading RoDLA model...")
        device = get_device()
        print(f"Using device: {device}")
        
        # Import custom modules
        from model.mmdet_custom.models.detectors import dino
        from model.mmdet_custom.models.dense_heads import dino_head
        
        _model_instance = init_detector(
            str(MODEL_CONFIG_PATH),
            str(MODEL_WEIGHTS_PATH),
            device=device
        )
        
        print("✅ RoDLA Model loaded successfully")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"GPU Memory: {allocated:.2f} GB allocated")
        
        return _model_instance
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise e


def get_model():
    """Get the loaded model instance"""
    if _model_instance is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model_instance


def get_model_classes():
    """Get the list of class names from the model"""
    model = get_model()
    return model.CLASSES