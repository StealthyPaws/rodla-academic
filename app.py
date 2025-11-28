#!/usr/bin/env python3
"""
RoDLA Document Layout Analysis - HuggingFace Spaces Entry Point
This file imports and runs the FastAPI application from the backend module.
"""

import sys
import os

# Add deployment directory to path to import backend modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deployment', 'backend'))

# Import the FastAPI app from backend_amar
from backend_amar import app

# For HuggingFace Spaces - the app will be run by uvicorn
# uvicorn command: uvicorn app:app --host 0.0.0.0 --port 7860

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )
