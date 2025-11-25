"""
RoDLA Object Detection API - Refactored Main Backend
Clean separation of concerns with modular components
Now with Perturbation Support!
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

# Import configuration
from config.settings import (
    API_TITLE, API_HOST, API_PORT,
    CORS_ORIGINS, CORS_METHODS, CORS_HEADERS,
    OUTPUT_DIR, PERTURBATION_OUTPUT_DIR  # NEW
)

# Import core functionality
from core.model_loader import load_model

# Import API routes
from api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description="RoDLA Document Layout Analysis API with comprehensive metrics and perturbation testing",
    version="2.1.0"  # Bumped version for perturbation feature
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize model and create directories on startup"""
    try:
        print("="*60)
        print("Starting RoDLA Document Layout Analysis API")
        print("="*60)
        
        # Create output directories
        print("📁 Creating output directories...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PERTURBATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Main output: {OUTPUT_DIR}")
        print(f"   ✓ Perturbations: {PERTURBATION_OUTPUT_DIR}")
        
        # Load model
        print("\n🔧 Loading RoDLA model...")
        load_model()
        
        print("\n" + "="*60)
        print("✅ API Ready!")
        print("="*60)
        print(f"🌐 Main API: http://{API_HOST}:{API_PORT}")
        print(f"📚 Docs: http://{API_HOST}:{API_PORT}/docs")
        print(f"📖 ReDoc: http://{API_HOST}:{API_PORT}/redoc")
        print("\n🎯 Available Endpoints:")
        print("   • GET  /api/model-info              - Model information")
        print("   • POST /api/detect                  - Standard detection")
        print("   • GET  /api/perturbations/info      - Perturbation info (NEW)")
        print("   • POST /api/perturb                 - Apply perturbations (NEW)")
        print("   • POST /api/detect-with-perturbation - Detect with perturbations (NEW)")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "="*60)
    print("🛑 Shutting down RoDLA API...")
    print("="*60)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )