import asyncio
import os
import glob
import tempfile
import shutil
from services.batch_job_manager import get_job_manager
from services.batch_processing import process_batch_job
from core.model_loader import load_model

# Load the model first
load_model()

async def test_batch():
    # 1. Define the directory path
    dataset_dir = "/mnt/d/MyStuff/University/Current/CV/Project/RoDLA/datasets/M6Doc-P/M6Doc-P/Speckle/Speckle_Easy/val2017"
    
    # 2. Dynamic lookup: Find all .jpg files in the directory
    # We look for both lowercase and uppercase extensions
    available_images = glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.JPG"))
    
    # Sort them to ensure deterministic order (optional but good for testing)
    available_images.sort()

    # 3. Validation
    if not available_images:
        print(f"❌ Error: No images found in {dataset_dir}")
        return

    # Take top 3
    src_images = available_images[:3]
    print(f"✅ Found {len(src_images)} images for testing: {[os.path.basename(x) for x in src_images]}")

    temp_files = []
    try:
        # Copy to temp files to simulate batch
        for src in src_images:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            shutil.copy(src, tmp.name)
            temp_files.append(tmp.name)
        
        # Create job manager
        manager = get_job_manager()
        config = {
            "total_images": len(temp_files),
            "score_threshold": 0.3,
            "visualization_mode": "summary",
            "perturbations": None,
            "perturbation_mode": "shared",
            "save_json": True,
            "filenames": [f"test_{os.path.basename(src)}" for src in src_images]
        }

        # Create job
        job_id = manager.create_job(config)
        print(f"Created job: {job_id}")

        # Process batch
        await process_batch_job(job_id, temp_files, config, manager)

        # Check results
        job = manager.get_job(job_id)
        print(f"\nFinal status: {job['status']}")
        print(f"Successful: {job['progress']['successful']}")
        print(f"Failed: {job['progress']['failed']}")
        print(f"Output dir: {job['output_dir']}")

    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass

# Run test
if __name__ == "__main__":
    asyncio.run(test_batch())