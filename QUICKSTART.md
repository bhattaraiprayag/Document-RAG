# Quickstart Guide

This guide will help you get the Document-RAG system up and running on your local machine.

## Prerequisites

-   **Docker Desktop**: Ensure Docker and Docker Compose are installed and running.
-   **Generative AI API Key**: You need an API key from a supported provider (e.g., OpenAI, Google Gemini).
-   **uv**: (Optional, for local non-Docker development) An extremely fast Python package manager. [Install uv](https://github.com/astral-sh/uv).
-   **Node.js**: (Optional, for local Frontend development) Version 18+ recommended.

## Quick Setup (Docker)

This is the recommended way to run the application.

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd Document-RAG
    ```

2.  **Configure Environment**:
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Open `.env` and fill in your API key (e.g., `OPENAI_API_KEY` or `GEMINI_API_KEY`).

3.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```
    -   The first run might take a few minutes to download base images and ML models.
    -   Once running, access the app at `http://localhost:3000`.

## Local Development (Without Docker)

If you prefer to run services individually for development/debugging, follow these steps.

### 1. Database (Qdrant)
You must run Qdrant via Docker as it's the easiest way.
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant
```

### 2. ML-API (Python/uv)
The ML-API handles embeddings and reranking. It requires PyTorch.

1.  Navigate to the directory:
    ```bash
    cd ml-api
    ```
2.  Install dependencies (this will create a virtual environment in `.venv`):
    ```bash
    uv sync
    ```
    *Note: On Linux, this will install CUDA-enabled PyTorch. On Windows/Mac, it will install the CPU version by default unless configured otherwise.*
3.  Run the service:
    ```bash
    uv run ml_api.py
    ```
    The service will start on `http://localhost:8001`.

### 3. Backend (Python/uv)
The orchestration layer.

1.  Navigate to the directory:
    ```bash
    cd backend
    ```
2.  Install dependencies:
    ```bash
    uv sync
    ```
3.  Run the service:
    ```bash
    uv run uvicorn app.main:app --reload --port 8000
    ```
    The service will start on `http://localhost:8000`.

### 4. Frontend (Node.js)
1.  Navigate to the directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    pnpm install
    ```
3.  Start the development server:
    ```bash
    pnpm run dev
    ```
    Access the frontend at `http://localhost:5173`.

## 5. Pre-commit Hooks (Git Hygiene)

This project uses `pre-commit` to ensure code quality.

1.  Navigate to the root directory.
2.  Install the hooks:
    ```bash
    # Ensure you have 'pre-commit' installed (installed via backend uv sync usually, or pip install pre-commit)
    cd backend
    uv run pre-commit install
    ```
    *Note: `pre-commit` is listed in backend dev-dependencies.*

## Troubleshooting

-   **CUDA Errors**: If running `ml-api` locally on a GPU machine, ensure you have the correct NVIDIA drivers installed. `uv` handles the Python-side CUDA libraries, but system drivers are required.
-   **Connection Refused**: Ensure all services (especially Qdrant on port 6333) are running.
-   **Build Failures**: If Docker build fails, try pruning your docker builder cache: `docker builder prune`.
