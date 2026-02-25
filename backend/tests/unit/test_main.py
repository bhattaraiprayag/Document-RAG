"""Unit tests for FastAPI endpoints in app.main."""

import importlib
import sys
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient


class DummyChunkingEngine:
    """Lightweight chunker stub for app import."""

    def __init__(self, *args, **kwargs) -> None:
        pass


class DummyDB:
    """In-memory DB stub for endpoint tests."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def file_exists(self, file_hash: str) -> bool:
        return file_hash == "indexed-file"

    def get_all_files(self) -> list[dict[str, str]]:
        return [{"file_hash": "indexed-file", "file_name": "demo.md"}]

    def delete_file(self, file_hash: str) -> None:
        self.deleted.append(file_hash)


class DummyRAG:
    """RAG stub with deterministic stream output."""

    def __init__(self) -> None:
        self.db = DummyDB()

    async def query(self, query: str, selected_files=None, chat_history=None):
        yield f"echo:{query}"


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    """Import app.main with lightweight dependencies patched in."""
    import app.chunking.engine as chunking_engine_module
    import app.database.qdrant_client as qdrant_client_module
    import app.rag.orchestrator as orchestrator_module

    monkeypatch.setattr(chunking_engine_module, "ChunkingEngine", DummyChunkingEngine)
    monkeypatch.setattr(qdrant_client_module, "QdrantDB", DummyDB)
    monkeypatch.setattr(orchestrator_module, "RAGOrchestrator", DummyRAG)

    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    main = importlib.import_module("app.main")

    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    main.UPLOAD_DIR.mkdir(exist_ok=True)

    main.ingestion_manager.start = AsyncMock()
    main.ingestion_manager.stop = AsyncMock()
    main.ingestion_manager.add_job = AsyncMock(return_value="queued-file")
    main.ingestion_manager.get_status = Mock(return_value=None)

    return main


@pytest.fixture
def client(main_module):
    """Test client with startup/shutdown enabled."""
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Health endpoint should return service metadata."""
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "services" in body
    assert "qdrant" in body["services"]


def test_list_documents_endpoint(client):
    """List documents should return DB-backed file list."""
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == [{"file_hash": "indexed-file", "file_name": "demo.md"}]


def test_get_status_returns_complete_for_indexed_file(client):
    """Unknown queue item returns complete when file already exists in DB."""
    response = client.get("/api/documents/status/indexed-file")

    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "complete"
    assert body["progress"] == 1.0


def test_upload_rejects_unsupported_extension(client):
    """Upload should reject unsupported file extensions."""
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.csv", b"bad format", "text/csv")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_chat_stream_endpoint_returns_done_marker(client):
    """Streaming endpoint should include final done marker."""
    response = client.post("/api/chat/stream", json={"query": "hello"})

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
