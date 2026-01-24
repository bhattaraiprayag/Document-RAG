# Document-RAG

A modern, containerized, end-to-end Retrieval-Augmented Generation (RAG) system for document Q&A.

## Motivation

Building a robust RAG system involves more than just a script. It requires:
-   **Reliable Ingestion**: Handling file uploads and chunking them intelligently.
-   **High-Quality Retrieval**: Using state-of-the-art embedding models (`bge-m3`) and vector databases (Qdrant).
-   **Precision**: Re-ranking results (`bge-reranker`) to ensure the LLM gets the *best* context, reducing hallucinations.
-   **Scalability**: Decoupling the heavy ML inference from the lightweight application logic.

This project demonstrates a production-ready architecture for such a system.

## Project Structure

-   `backend/`: FastAPI application for orchestration. Managed with `uv`.
-   `ml-api/`: Dedicated microservice for Embeddings and Reranking. Managed with `uv`.
-   `frontend/`: React/Vite/Tailwind UI.
-   `models_cache/`: Shared volume for storing downloaded ML models.
-   `qdrant_data/`: Persistent storage for the vector database.
-   `uploads/`: Storage for uploaded documents.

## Documentation

-   [**Quickstart Guide**](QUICKSTART.md): Learn how to set up and run the system (Docker & Local).
-   [**Architecture**](ARCHITECTURE.md): Deep dive into the system design, data flow, and stack choices.

## Key Features

-   **Modern Stack**: Python 3.10+, React 18, FastAPI, Docker.
-   **Efficient Dependency Management**: Uses **uv** for lightning-fast, reproducible Python environments.
-   **GPU Acceleration**: `ml-api` is optimized for CUDA but degrades gracefully to CPU.
-   **Interactive UI**: Clean, responsive chat interface.

## License

[MIT](LICENSE)
