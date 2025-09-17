#!/bin/bash
# Script to start Docker Compose with environment variables from backend/.env

# Load all environment variables from backend/.env
set -a  # automatically export all variables
source backend/.env
set +a  # turn off automatic export

# Start Docker Compose
docker compose up -d

# Show status
docker compose ps