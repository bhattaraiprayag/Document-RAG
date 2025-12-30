# =============================================================================
# Stage 1: Builder - Install dependencies with uv
# =============================================================================
FROM python:3.10-slim AS builder

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Set uv environment variables for optimal caching
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Copy project files for dependency installation
COPY pyproject.toml uv.lock* ./

# Install dependencies (cached layer - this is the slow part)
# Using --frozen to skip lockfile validation when uv.lock exists
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# Copy application code
COPY ml_api.py ./

# Final sync to install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable


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

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

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
