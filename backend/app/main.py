"""FastAPI main application."""

import os  # noqa: E402

# Suppress HuggingFace symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # noqa: E402

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from markitdown import MarkItDown
from pydantic import BaseModel

from .chunking.engine import ChunkingEngine
from .config import settings
from .database.qdrant_client import QdrantDB
from .observability.metrics import metrics
from .rag.orchestrator import RAGOrchestrator
from .utils.batch_embed import DEFAULT_EMBED_BATCH_SIZE, embed_texts_in_batches

# Configuration
UPLOAD_DIR = Path("./uploads")
EMBED_BATCH_SIZE = DEFAULT_EMBED_BATCH_SIZE
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".epub", ".txt", ".md"}


# --- Ingestion Queue ---
class IngestionStage(str, Enum):
    """Ingestion pipeline stages."""

    QUEUED = "queued"
    HASHING = "hashing"
    CONVERTING = "converting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class IngestionJob:
    """Represents a document ingestion job."""

    file_hash: str
    file_name: str
    file_path: str
    stage: IngestionStage = IngestionStage.QUEUED
    progress: float = 0.0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class IngestionManager:
    """Manages document ingestion queue and processing."""

    def __init__(self) -> None:
        """Initialize ingestion manager."""
        self.jobs: dict[str, IngestionJob] = {}
        self.queue: asyncio.Queue[IngestionJob] = asyncio.Queue()
        self.chunker = ChunkingEngine()
        self.db = QdrantDB()
        self.markitdown = MarkItDown()
        self._worker_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the ingestion worker."""
        self._worker_task = asyncio.create_task(self._worker())
        print("✅ Ingestion worker started.")

    async def stop(self) -> None:
        """Stop the ingestion worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def add_job(self, file_path: str, file_name: str) -> str:
        """
        Add a file to the ingestion queue.

        Args:
            file_path: Path to the file
            file_name: Original filename

        Returns:
            File hash (SHA256)
        """
        # Compute hash
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Check if already indexed
        if self.db.file_exists(file_hash):
            print(f"ℹ️  File already indexed: {file_name} ({file_hash[:8]}...)")
            return file_hash

        # Check if already processing
        if file_hash in self.jobs:
            print(f"ℹ️  File already in queue: {file_name} ({file_hash[:8]}...)")
            return file_hash

        job = IngestionJob(
            file_hash=file_hash, file_name=file_name, file_path=file_path
        )
        self.jobs[file_hash] = job
        await self.queue.put(job)
        print(f"📄 Queued for ingestion: {file_name} ({file_hash[:8]}...)")

        return file_hash

    def get_status(self, file_hash: str) -> Optional[IngestionJob]:
        """Get ingestion status for a file."""
        return self.jobs.get(file_hash)

    async def _worker(self) -> None:
        """Worker that processes ingestion jobs."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            while True:
                try:
                    job = await self.queue.get()
                    await self._process_job(job, client)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"❌ Worker error: {e}")

    async def _process_job(self, job: IngestionJob, client: Any) -> None:
        """Process a single ingestion job."""
        try:
            print(f"\n{'=' * 60}")
            print(f"🔄 Processing: {job.file_name}")
            print(f"{'=' * 60}")

            # Stage: Converting
            job.stage = IngestionStage.CONVERTING
            job.progress = 0.2
            metrics.increment("ingestion_started")
            print("📝 Converting to markdown...")

            result = self.markitdown.convert(job.file_path)
            markdown_content = result.text_content
            print(f"✅ Converted ({len(markdown_content)} chars)")

            # Stage: Chunking
            job.stage = IngestionStage.CHUNKING
            job.progress = 0.4
            print("✂️  Chunking document...")

            parents, children = self.chunker.chunk_document(
                markdown_content, job.file_hash, job.file_name
            )
            print(f"✅ Created {len(parents)} parents, {len(children)} children")

            # Stage: Embedding
            job.stage = IngestionStage.EMBEDDING
            job.progress = 0.5
            total_chunks = len(children)
            total_batches = (total_chunks + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
            print(f"🧮 Embedding {total_chunks} chunks in {total_batches} batches...")

            # Progress callback for granular tracking
            def on_batch_progress(current: int, total: int) -> None:
                # Progress from 0.5 to 0.8 during embedding
                batch_progress = current / total
                job.progress = 0.5 + (batch_progress * 0.3)
                print(f"   📦 Batch {current}/{total} complete")

            # Embed children in controlled batches to prevent memory explosion
            child_texts = [c.content for c in children]
            dense_vecs, sparse_vecs, total_latency = await embed_texts_in_batches(
                client=client,
                texts=child_texts,
                embed_api_url=settings.embed_api_url,
                batch_size=EMBED_BATCH_SIZE,
                progress_callback=on_batch_progress,
            )
            print(f"✅ Embeddings generated ({total_latency:.1f}ms total)")

            # Stage: Indexing
            job.stage = IngestionStage.INDEXING
            job.progress = 0.8
            print("💾 Indexing to Qdrant...")

            # Store in Qdrant
            self.db.store_chunks(
                chunks=[
                    {
                        "id": c.id,
                        "content": c.content,
                        "parent_id": c.parent_id,
                        "file_hash": c.file_hash,
                        "file_name": c.file_name,
                        "chunk_index": c.chunk_index,
                        "header_path": c.header_path,
                    }
                    for c in children
                ],
                dense_vectors=dense_vecs,
                sparse_vectors=sparse_vecs,
            )

            self.db.store_parents(
                [
                    {
                        "id": p.id,
                        "content": p.content,
                        "file_hash": p.file_hash,
                        "file_name": p.file_name,
                        "header_path": p.header_path,
                        "child_ids": p.child_ids,
                    }
                    for p in parents
                ]
            )

            # Complete
            job.stage = IngestionStage.COMPLETE
            job.progress = 1.0
            metrics.increment("ingestion_completed")
            print(f"✅ COMPLETE: {job.file_name}")
            print(f"{'=' * 60}\n")

        except Exception as e:
            job.stage = IngestionStage.FAILED
            job.error = str(e)
            metrics.increment("ingestion_failed")
            print(f"❌ Ingestion failed for {job.file_name}: {e}")


# --- Application Lifespan ---
ingestion_manager = IngestionManager()
rag = RAGOrchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan manager."""
    print("\n" + "=" * 60)
    print("🚀 Starting Hybrid RAG Backend")
    print("=" * 60)
    await ingestion_manager.start()
    print("✅ All systems operational")
    print("=" * 60 + "\n")

    yield

    print("\n🛑 Shutting down...")
    await ingestion_manager.stop()
    print("✅ Shutdown complete\n")


app = FastAPI(
    title="Hybrid RAG API",
    description="Local, zero-cost RAG system with parent-child chunking",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",  # Vite default ports
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    """Chat request model."""

    query: str
    selected_files: Optional[list[str]] = None
    chat_history: Optional[list[dict[str, str]]] = None


class FileInfo(BaseModel):
    """File information model."""

    file_hash: str
    file_name: str


class IngestionStatus(BaseModel):
    """Ingestion status model."""

    file_hash: str
    file_name: str
    stage: str
    progress: float
    error: Optional[str] = None


# --- Endpoints ---
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
    """Upload a document for ingestion."""
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Save file
    file_path = UPLOAD_DIR / (file.filename or "upload")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Queue for ingestion
    file_hash = await ingestion_manager.add_job(
        str(file_path), file.filename or "upload"
    )

    return {"file_hash": file_hash, "message": "Queued for ingestion"}


@app.get("/api/documents/status/{file_hash}", response_model=IngestionStatus)
async def get_ingestion_status(file_hash: str) -> IngestionStatus:
    """Get ingestion status for a file."""
    job = ingestion_manager.get_status(file_hash)
    if not job:
        # Check if already in database
        if rag.db.file_exists(file_hash):
            return IngestionStatus(
                file_hash=file_hash,
                file_name="Unknown",
                stage="complete",
                progress=1.0,
            )
        raise HTTPException(404, "File not found in queue")

    return IngestionStatus(
        file_hash=job.file_hash,
        file_name=job.file_name,
        stage=job.stage.value,
        progress=job.progress,
        error=job.error,
    )


@app.get("/api/documents", response_model=list[FileInfo])
async def list_documents() -> list[FileInfo]:
    """List all indexed documents."""
    files = rag.db.get_all_files()
    return [FileInfo(**f) for f in files]


@app.delete("/api/documents/{file_hash}")
async def delete_document(file_hash: str) -> dict[str, str]:
    """Delete a document and all its chunks."""
    rag.db.delete_file(file_hash)
    metrics.increment("documents_deleted")
    return {"message": "Document deleted"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a chat response using SSE."""

    async def generate() -> Any:
        try:
            async for token in rag.query(
                request.query, request.selected_files, request.chat_history
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback

            error_type = type(e).__name__
            error_msg = str(e) if str(e) else "Unknown error"
            full_error = f"{error_type}: {error_msg}"
            print(f"❌ Chat error: {full_error}")
            traceback.print_exc()
            yield f"data: {json.dumps({'error': full_error})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0",
        "services": {
            "qdrant": settings.qdrant_url,
            "ml_api": settings.embed_api_url,  # Unified ML API (embed + rerank)
            "ollama": settings.ollama_base_url,
        },
        "metrics": metrics.report(),
    }


@app.get("/api/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get system metrics."""
    return metrics.report()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
