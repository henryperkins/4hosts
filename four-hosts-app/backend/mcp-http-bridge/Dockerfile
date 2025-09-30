# HTTP-to-Stdio Bridge for Azure AI Foundry MCP Server
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bridge server
COPY bridge_server.py .

# Set environment variables
ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=8081
ENV PYTHONUNBUFFERED=1

# Pre-download the MCP server to avoid doing it at runtime
# This caches the git clone and dependencies
RUN uvx --prerelease=allow --from git+https://github.com/azure-ai-foundry/mcp-foundry.git run-azure-ai-foundry-mcp --help || true

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the bridge server
CMD ["python", "bridge_server.py"]