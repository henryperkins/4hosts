# Multi-stage Dockerfile for Four Hosts Backend
# Using alternative registry to avoid Docker Hub authentication issues
FROM public.ecr.aws/docker/library/python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# --- Production Stage ---
FROM public.ecr.aws/docker/library/python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgomp1 \
    libopenblas0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set explicit HOME for appuser and prepare HuggingFace cache directories
ENV HOME=/home/appuser
RUN mkdir -p /home/appuser/.cache/huggingface/hub && \
    chown -R appuser:appuser /home/appuser

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/*

# Copy application code
COPY --chown=appuser:appuser . .

# Copy entrypoint script and ensure it is executable
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && chown appuser:appuser /app/docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/nltk_data /app/model_cache && \
    chown -R appuser:appuser /app

# NOTE: NLTK data and HuggingFace models are NOT downloaded at build time
# to keep image size small. They will be downloaded at first runtime or
# should be mounted from a persistent volume.
#
# If you prefer to bake them into the image (increases size by ~8-10GB):
# RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); \
#     nltk.download('stopwords', download_dir='/app/nltk_data'); \
#     nltk.download('wordnet', download_dir='/app/nltk_data')" && \
#     chown -R appuser:appuser /app/nltk_data

# Ensure ownership (defensive) of HF cache after code copy
RUN chown -R appuser:appuser /home/appuser/.cache

# Configure runtime environment
ENV NLTK_DATA=/app/nltk_data
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Explicitly set HF and model cache locations
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TORCH_HOME=/app/model_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run migrations before launching the app
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command launches uvicorn (can be overridden at runtime)
CMD ["uvicorn", "main_new:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
