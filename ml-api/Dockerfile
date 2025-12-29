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
# Stage 2: Runtime - Minimal image with only what's needed
# =============================================================================
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

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
