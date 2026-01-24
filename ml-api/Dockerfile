# Stage 1: Builder
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Explicitly tell uv to use the system python
ENV UV_PYTHON_PREFERENCE=system


COPY pyproject.toml uv.lock ./


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


RUN .venv/bin/python -c "import fastapi; print('FastAPI installed successfully')"

# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY --from=builder /app/.venv /app/.venv


ENV PATH="/app/.venv/bin:$PATH"


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY ml_api.py ./
RUN mkdir -p /models_cache

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8001/health || exit 1

# Explicitly use the venv python
CMD ["/app/.venv/bin/python", "ml_api.py"]
