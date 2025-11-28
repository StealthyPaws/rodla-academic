"""
Register DINO detector with MMDET if not already registered
This allows loading RoDLA models without requiring DCNv3 compilation
"""

import sys
from pathlib import Path

def register_dino():
    """Register DINO with MMDET model registry"""
    try:
        from mmdet.models.builder import DETECTORS, BACKBONES, NECKS, HEADS
        
        # Check if already registered
        if 'DINO' in DETECTORS.module_dict:
            print("✅ DINO already registered in MMDET")
            return True
        
        print("⏳ Registering DINO detector...")
        
        # Try to import and register custom models
        # Use absolute path from /home/admin/CV/rodla-academic
        REPO_ROOT = Path("/home/admin/CV/rodla-academic")
        sys.path.insert(0, str(REPO_ROOT / "model"))
        sys.path.insert(0, str(REPO_ROOT / "model" / "ops_dcnv3"))
        
        try:
            import mmdet_custom
            if 'DINO' in DETECTORS.module_dict:
                print("✅ DINO registered successfully from mmdet_custom")
                return True
            else:
                print("⚠️  DINO not found in mmdet_custom registry")
                return False
        except ModuleNotFoundError as e:
            if "DCNv3" in str(e):
                print(f"⚠️  Cannot register DINO: DCNv3 module not available")
                print(f"   Error: {e}")
                return False
            else:
                print(f"❌ Error importing mmdet_custom: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error registering DINO: {e}")
        return False


def try_load_with_dino_registration(config_path: str, checkpoint_path: str, device: str = "cpu"):
    """Try to load a DINO model, registering it if necessary"""
    from mmdet.apis import init_detector
    
    # Try registering DINO first
    dino_registered = register_dino()
    
    if not dino_registered:
        print("⚠️  DINO could not be registered")
        print("   Will attempt to load anyway...")
    
    # Try to load the model
    try:
        print(f"⏳ Loading model from {checkpoint_path}...")
        model = init_detector(config_path, checkpoint_path, device=device)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
