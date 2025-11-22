"""
RoDLA Object Detection API - Refactored Main Backend
Clean separation of concerns with modular components
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration
from config.settings import (
    API_TITLE, API_HOST, API_PORT,
    CORS_ORIGINS, CORS_METHODS, CORS_HEADERS
)

# Import core functionality
from core.model_loader import load_model

# Import API routes
from api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description="RoDLA Document Layout Analysis API with comprehensive metrics",
    version="2.0.0"
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
    """Initialize model on startup"""
    try:
        print("="*60)
        print("Starting RoDLA Document Layout Analysis API")
        print("="*60)
        load_model()
        print("="*60)
        print("API Ready!")
        print(f"Access at: http://{API_HOST}:{API_PORT}")
        print(f"Docs at: http://{API_HOST}:{API_PORT}/docs")
        print("="*60)
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nShutting down RoDLA API...")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )