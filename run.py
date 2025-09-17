#!/usr/bin/env python3
"""
Sales Follow-Up Assistant - Application Entry Point
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    reload = os.getenv("RELOAD", "false").lower() == "true"

    print(f"🚀 Starting Sales Follow-Up Assistant API on {host}:{port}")
    print(f"📊 Log Level: {log_level.upper()}")
    print(f"🔄 Auto-reload: {reload}")
    print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")

    # Run the FastAPI application
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        access_log=True
    )