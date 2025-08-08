#!/bin/bash
# Start the backend with a single worker to avoid Prometheus metrics conflicts
cd /home/azureuser/4hosts/four-hosts-app/backend
source venv/bin/activate
export ENVIRONMENT=development
python -m uvicorn main_new:app --host 0.0.0.0 --port 8000 --workers 1 --reload
