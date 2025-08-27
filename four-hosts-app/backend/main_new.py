#!/usr/bin/env python3
"""
Four Hosts Research API - Refactored Main Application
"""

import os
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.app import create_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    # Configure based on environment
    environment = os.getenv("ENVIRONMENT", "production")

    if environment == "production":
        # Production configuration - SINGLE WORKER until Redis is implemented
        uvicorn.run(
            "main_new:app",
            host="0.0.0.0",
            port=8000,
            workers=1,  # Changed from 4 to 1
            log_level="info",
            access_log=True,
            reload=False,
            server_header=False,
            date_header=False,
        )
    else:
        # Development configuration
        uvicorn.run(
            "main_new:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="debug"
        )


# Create app instance for uvicorn
app = create_app()

if __name__ == "__main__":
    main()
