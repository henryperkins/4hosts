#!/bin/bash
# Start the backend with a single worker to avoid Prometheus metrics conflicts
set -euo pipefail
cd /home/azureuser/4hosts/four-hosts-app/backend
source venv/bin/activate

# Development defaults: verbose logging and safer startup
export ENVIRONMENT=${ENVIRONMENT:-development}
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}
# In dev, allow startup without configured search providers
export SEARCH_DISABLE_FAILFAST=${SEARCH_DISABLE_FAILFAST:-1}

# Optional init timeouts to avoid indefinite hangs (override as needed)
# export INIT_DB_TIMEOUT_SEC=30
# export INIT_RESEARCH_STORE_TIMEOUT_SEC=20
# export INIT_CACHE_TIMEOUT_SEC=20
# export INIT_ORCHESTRATOR_TIMEOUT_SEC=90
# export INIT_LLM_TIMEOUT_SEC=30
# export INIT_SEARCH_MANAGER_TIMEOUT_SEC=30
# export INIT_HF_CLASSIFIER_TIMEOUT_SEC=20

python -m uvicorn main_new:app --host 0.0.0.0 --port 8000 --workers 1 --reload --log-level debug
