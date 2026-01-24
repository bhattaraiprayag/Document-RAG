# Stage 1: Builder
FROM python:3.10-slim AS builder


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy


COPY pyproject.toml uv.lock ./


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


COPY . .

# Sync the project environment (ensures all dependencies are in .venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app


COPY --from=builder /app/.venv /app/.venv


ENV PATH="/app/.venv/bin:$PATH"


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download NLTK data using the specific python from venv
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"


COPY app/ ./app/


RUN mkdir -p /app/uploads


EXPOSE 8000


HEALTHCHECK --interval=15s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/api/health || exit 1


CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000"]
