#!/bin/bash

# Docker Hub Authentication Workaround Script
# This script attempts to resolve Docker Hub authentication issues

echo "Attempting to fix Docker Hub authentication issues..."

# Clear any existing Docker credentials
echo "Clearing existing Docker credentials..."
docker logout 2>/dev/null || true

# Remove existing config
rm -f ~/.docker/config.json

# Create a new config with anonymous access
mkdir -p ~/.docker
cat > ~/.docker/config.json <<EOF
{
  "auths": {
    "https://index.docker.io/v1/": {}
  },
  "HttpHeaders": {
    "User-Agent": "Docker-Client/20.10.0"
  }
}
EOF

echo "Docker config reset for anonymous access."

# Try to pull images individually without authentication
echo "Attempting to pull base images..."

# Pull images with retry logic
pull_with_retry() {
    local image=$1
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "Attempting to pull $image (attempt $((retry_count + 1))/$max_retries)..."
        if docker pull $image; then
            echo "Successfully pulled $image"
            return 0
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "Failed to pull $image. Retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    echo "Failed to pull $image after $max_retries attempts"
    return 1
}

# Try pulling the required images
pull_with_retry "docker.io/library/postgres:15-alpine"
pull_with_retry "docker.io/library/redis:7-alpine"
pull_with_retry "docker.io/library/python:3.12-slim"
pull_with_retry "docker.io/library/node:22-alpine"
pull_with_retry "docker.io/library/nginx:alpine"

echo "Image pull attempts completed."
echo ""
echo "Now trying to build and run the containers..."
docker compose up -d --build