"""Model loading and initialization"""
from typing import Optional
import torch
import gc
from pathlib import Path
from mmdet.apis import init_detector
from config.settings import MODEL_CONFIG_PATH, MODEL_WEIGHTS_PATH

_model_instance: Optional[object] = None


def get_device() -> str:
    """Get the best available device"""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_weights_exist():
    """Ensure model weights exist, download if needed"""
    weights_path = Path(MODEL_WEIGHTS_PATH)
    
    # Check if weights exist and are valid (> 1MB)
    if weights_path.exists() and weights_path.stat().st_size > 1000000:
        print(f"✅ Model weights found: {weights_path}")
        print(f"   Size: {weights_path.stat().st_size / (1024**3):.2f} GB")
        return True
    
    print(f"⚠️  Model weights not found or invalid at: {weights_path}")
    print(f"   Attempting to download from Google Drive...")
    
    try:
        # Try to run download script
        from deployment.backend.download_weights import download_weights
        success = download_weights()
        if success:
            return True
    except Exception as e:
        print(f"⚠️  Could not auto-download weights: {e}")
    
    print(f"\n❌ Model weights required but not available!")
    print(f"   Please download from: https://drive.google.com/file/d/1BHyz2jH52Irt6izCeTRb4g2J5lXsA9cz/view?usp=sharing")
    print(f"   Place at: {weights_path}")
    return False


def load_model():
    """Load the RoDLA model with proper memory management"""
    global _model_instance
    
    if _model_instance is not None:
        return _model_instance
    
    try:
        # Ensure weights exist
        if not ensure_weights_exist():
            raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")
        
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