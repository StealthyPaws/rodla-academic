# Test script: test_job_manager.py
from services.batch_job_manager import get_job_manager

# Get manager instance
manager = get_job_manager()

# Create a test job
config = {
    "total_images": 5,
    "score_threshold": 0.3,
    "visualization_mode": "none",
    "filenames": ["test1.jpg", "test2.jpg", "test3.jpg"]
}

job_id = manager.create_job(config)
print(f"Created job: {job_id}")

# Update progress
manager.mark_job_started(job_id)
manager.update_job_progress(job_id, 1, 5, processing_time=2.3)
manager.add_job_result(job_id, "test1.jpg", {
    "status": "success",
    "processing_time": 2.3,
    "detections_count": 47,
    "average_confidence": 0.82
})

# Get job status
job = manager.get_job(job_id)
print(f"Job status: {job['status']}")
print(f"Progress: {job['progress']}")