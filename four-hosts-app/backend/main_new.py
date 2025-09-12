#!/usr/bin/env python3
"""
Four Hosts Research API - Refactored Main Application
"""

import os
import logging
import uvicorn
from dotenv import load_dotenv
from core.app import create_app

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    # Configure based on environment
    environment = os.getenv("ENVIRONMENT", "production")

    # ───────────────────────────────────────────────────────────
    #  Determine worker count
    #  • Default to 4 workers when Redis is configured (enables
    #    shared state across workers for cache & rate-limiter)
    #  • Allow override via WORKERS env var
    #  • Fallback to single-worker when Redis is unavailable
    # ───────────────────────────────────────────────────────────
    workers_cfg = int(os.getenv("WORKERS", "0") or 0)
    if workers_cfg > 0:
        workers = workers_cfg
    else:
        workers = 4 if os.getenv("REDIS_URL") else 1

    if environment == "production":
        uvicorn.run(
            "main_new:app",
            host="0.0.0.0",
            port=8000,
            workers=workers,
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
