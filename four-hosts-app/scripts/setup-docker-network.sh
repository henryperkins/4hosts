#!/usr/bin/env bash
set -euo pipefail

NETWORK_NAME=${1:-fourhosts-network}

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not on PATH." >&2
  exit 1
fi

if docker network ls --format '{{.Name}}' | grep -Fxq "$NETWORK_NAME"; then
  echo "Docker network '$NETWORK_NAME' already exists."
  exit 0
fi

echo "Creating docker network '$NETWORK_NAME'..."
docker network create "$NETWORK_NAME"
echo "Docker network '$NETWORK_NAME' created successfully."
