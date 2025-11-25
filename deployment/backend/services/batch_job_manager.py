"""
Batch Job Manager Service
==========================
Thread-safe manager for batch processing jobs with persistence.
"""

import threading
import json
import uuid
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import traceback


class BatchJobManager:
    """
    Thread-safe manager for batch processing jobs.
    
    Responsibilities:
    - Create and track batch jobs
    - Update job progress and results
    - Persist jobs to disk for recovery
    - Load existing jobs on startup
    
    All operations are thread-safe using locks.
    """
    
    def __init__(self, jobs_dir: Path):
        """
        Initialize the batch job manager.
        
        Args:
            jobs_dir: Directory to store job metadata files
        """
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory job registry
        self._jobs: Dict[str, dict] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load existing jobs from disk
        self._load_jobs_from_disk()
    
    def create_job(self, config: dict) -> str:
        """
        Create a new batch processing job.
        
        Args:
            config: Job configuration dictionary containing:
                - total_images: Number of images to process
                - score_threshold: Confidence threshold
                - visualization_mode: Visualization generation mode
                - perturbations: Optional perturbation configurations
                - perturbation_mode: "shared" or "per_image"
                - save_json: Whether to save results
                - filenames: List of original filenames
        
        Returns:
            Unique job ID (UUID4 string)
        """
        with self._lock:
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Create job structure
            now = datetime.now().isoformat()
            job = {
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
                
                "config": config,
                
                "progress": {
                    "current": 0,
                    "total": config["total_images"],
                    "percentage": 0.0,
                    "successful": 0,
                    "failed": 0,
                    "processing_times": []
                },
                
                "results": [],
                "summary": None,
                "output_dir": None,
                "error": None
            }
            
            # Store in memory
            self._jobs[job_id] = job
            
            # Persist to disk
            self._persist_job(job_id)
            
            print(f"✅ Created batch job: {job_id}")
            print(f"   Total images: {config['total_images']}")
            print(f"   Visualization mode: {config['visualization_mode']}")
            
            return job_id
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            Job dictionary if found, None otherwise
        """
        with self._lock:
            job = self._jobs.get(job_id)
            
            # If not in memory, try loading from disk
            if job is None:
                job_file = self.jobs_dir / f"job_{job_id}.json"
                if job_file.exists():
                    try:
                        with open(job_file, 'r') as f:
                            job = json.load(f)
                        self._jobs[job_id] = job
                        print(f"📂 Loaded job {job_id} from disk")
                    except Exception as e:
                        print(f"❌ Error loading job {job_id}: {e}")
                        return None
            
            return job
    
    def mark_job_started(self, job_id: str):
        """
        Mark a job as started (processing).
        
        Args:
            job_id: Job identifier
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            job["status"] = "processing"
            job["started_at"] = datetime.now().isoformat()
            job["updated_at"] = datetime.now().isoformat()
            
            self._persist_job(job_id)
            print(f"▶️  Started processing job: {job_id}")
    
    def update_job_progress(
        self, 
        job_id: str, 
        current: int, 
        total: int,
        processing_time: Optional[float] = None
    ):
        """
        Update job progress after processing an image.
        
        Args:
            job_id: Job identifier
            current: Number of images processed so far
            total: Total number of images
            processing_time: Time taken for this image (seconds)
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            job["progress"]["current"] = current
            job["progress"]["total"] = total
            job["progress"]["percentage"] = round((current / total) * 100, 2)
            job["updated_at"] = datetime.now().isoformat()
            
            if processing_time is not None:
                job["progress"]["processing_times"].append(processing_time)
            
            # Persist to disk after each update
            self._persist_job(job_id)
            
            print(f"📊 Job {job_id}: {current}/{total} ({job['progress']['percentage']}%)")
    
    def add_job_result(
        self, 
        job_id: str, 
        image_name: str, 
        result: dict
    ):
        """
        Add result for a single image to the job.
        
        Args:
            job_id: Job identifier
            image_name: Original image filename
            result: Result dictionary with keys:
                - status: "success" or "failed"
                - processing_time: Time in seconds
                - detections_count: Number of detections (if success)
                - average_confidence: Average confidence (if success)
                - result_path: Path to saved result JSON (if success)
                - error: Error message (if failed)
                - traceback: Full traceback (if failed)
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            
            # Add image name to result
            result["image"] = image_name
            
            # Update success/failure counts
            if result["status"] == "success":
                job["progress"]["successful"] += 1
            elif result["status"] == "failed":
                job["progress"]["failed"] += 1
            
            # Add to results list
            job["results"].append(result)
            job["updated_at"] = datetime.now().isoformat()
            
            # Persist to disk
            self._persist_job(job_id)
            
            status_emoji = "✅" if result["status"] == "success" else "❌"
            print(f"{status_emoji} Image {image_name}: {result['status']}")
    
    def mark_job_completed(
        self, 
        job_id: str,
        summary: Optional[dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        Mark a job as completed.
        
        Args:
            job_id: Job identifier
            summary: Optional summary statistics
            output_dir: Path to batch output directory
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            
            # Determine final status
            total_images = job["progress"]["total"]
            successful = job["progress"]["successful"]
            failed = job["progress"]["failed"]
            
            if successful == total_images:
                job["status"] = "completed"
            elif successful > 0:
                job["status"] = "partial"
            else:
                job["status"] = "failed"
            
            job["completed_at"] = datetime.now().isoformat()
            job["updated_at"] = datetime.now().isoformat()
            
            if summary:
                job["summary"] = summary
            
            if output_dir:
                job["output_dir"] = output_dir
            
            self._persist_job(job_id)
            
            print(f"🏁 Job {job_id} completed: {job['status']}")
            print(f"   Successful: {successful}/{total_images}")
            print(f"   Failed: {failed}/{total_images}")
    
    def mark_job_failed(self, job_id: str, error: str):
        """
        Mark a job as failed due to critical error.
        
        Args:
            job_id: Job identifier
            error: Error message
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self._jobs[job_id]
            job["status"] = "failed"
            job["error"] = error
            job["updated_at"] = datetime.now().isoformat()
            
            if job["started_at"] and not job["completed_at"]:
                job["completed_at"] = datetime.now().isoformat()
            
            self._persist_job(job_id)
            
            print(f"❌ Job {job_id} failed: {error}")
    
    def _persist_job(self, job_id: str):
        """
        Save job to disk as JSON file.
        
        Args:
            job_id: Job identifier
        """
        if job_id not in self._jobs:
            return
        
        job_file = self.jobs_dir / f"job_{job_id}.json"
        
        try:
            with open(job_file, 'w') as f:
                json.dump(self._jobs[job_id], f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Failed to persist job {job_id}: {e}")
    
    def _load_jobs_from_disk(self):
        """
        Load all existing jobs from disk on startup.
        
        Jobs in "queued" or "processing" status are marked as failed
        with message "Server restarted during processing".
        """
        print(f"📂 Loading existing jobs from {self.jobs_dir}...")
        
        job_files = list(self.jobs_dir.glob("job_*.json"))
        loaded_count = 0
        
        for job_file in job_files:
            try:
                with open(job_file, 'r') as f:
                    job = json.load(f)
                
                job_id = job["job_id"]
                
                # Mark incomplete jobs as failed
                if job["status"] in ["queued", "processing"]:
                    job["status"] = "failed"
                    job["error"] = "Server restarted during processing"
                    job["updated_at"] = datetime.now().isoformat()
                    
                    # Re-save with updated status
                    with open(job_file, 'w') as f:
                        json.dump(job, f, indent=2)
                
                self._jobs[job_id] = job
                loaded_count += 1
                
            except Exception as e:
                print(f"⚠️  Warning: Error loading {job_file.name}: {e}")
                traceback.print_exc()
        
        print(f"✅ Loaded {loaded_count} jobs from disk")
        
        # Print summary
        if loaded_count > 0:
            statuses = {}
            for job in self._jobs.values():
                status = job["status"]
                statuses[status] = statuses.get(status, 0) + 1
            
            print("   Status summary:")
            for status, count in statuses.items():
                print(f"   - {status}: {count}")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_job_manager_instance: Optional[BatchJobManager] = None


def get_job_manager() -> BatchJobManager:
    """
    Get the singleton BatchJobManager instance.
    
    Returns:
        Singleton BatchJobManager instance
    """
    global _job_manager_instance
    
    if _job_manager_instance is None:
        from config.settings import JOBS_DIR
        _job_manager_instance = BatchJobManager(JOBS_DIR)
    
    return _job_manager_instance