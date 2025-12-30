# =============================================================================
# Stage 1: Builder - Export pinned requirements using uv
# =============================================================================
FROM python:3.10-slim AS builder

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./

# Export locked dependencies to requirements.txt for pip
# This gives us the speed of uv's resolver with pip's portability
RUN uv export --frozen --no-hashes --no-dev -o requirements.txt


# =============================================================================
# Stage 2: Runtime - NVIDIA CUDA image for GPU acceleration
# =============================================================================
# CRITICAL: Use NVIDIA CUDA runtime image for onnxruntime-gpu support
# This provides libcublasLt.so.12, cuDNN, and other CUDA libraries
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy exported requirements from builder
COPY --from=builder /app/requirements.txt ./

# Install dependencies via pip (compatible with this Python installation)
# Using --no-cache-dir to keep image size small
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Set CUDA-related environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy application code
COPY ml_api.py ./

# Create models cache directory
RUN mkdir -p /models_cache

# Expose single port for both endpoints
EXPOSE 8001

# Health check
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the unified ML API
CMD ["python", "ml_api.py"]
