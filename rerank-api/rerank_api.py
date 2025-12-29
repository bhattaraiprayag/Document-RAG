"""
BGE-Reranker-Base GPU API with FP16.
Cross-encoder reranking for RTX 3050 (4GB VRAM).
"""
import os
from pathlib import Path

# CRITICAL: Set cache directories BEFORE any HuggingFace imports
MODELS_CACHE_DIR = Path(__file__).parent.parent / "models_cache"
MODELS_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(MODELS_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_CACHE_DIR / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_CACHE_DIR / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print(f"🔧 HF_HOME set to: {MODELS_CACHE_DIR}")
print(f"🔧 All models will be cached here.")

# NOW import HuggingFace libraries (after env vars are set)
import time
import asyncio
from contextlib import asynccontextmanager
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from FlagEmbedding import FlagReranker


# Configuration
MODEL_ID = "BAAI/bge-reranker-base"
MAX_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
BATCH_TIMEOUT_S = float(os.getenv("RERANK_BATCH_TIMEOUT", "0.02"))
MAX_QUEUE_SIZE = 100

# Global resources
reranker = None


class RerankBatcher:
    """Async batching for reranking requests."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.processing_loop_task = None

    async def start(self) -> None:
        """Start the background processing loop."""
        self.processing_loop_task = asyncio.create_task(self._process_loop())
        print("Rerank batch processor started.")

    async def stop(self) -> None:
        """Stop the processing loop."""
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass

    async def process(self, query: str, documents: List[str]) -> dict:
        """
        Process a reranking request.

        Args:
            query: Query text
            documents: List of documents to rerank

        Returns:
            Dict with results, latency, and batch_size
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((query, documents, future))
        return await future

    async def _process_loop(self) -> None:
        """Background loop that collects and processes batches."""
        while True:
            batch_data = []

            # Wait for first request
            try:
                item = await self.queue.get()
                batch_data.append(item)
            except asyncio.CancelledError:
                break

            # Collect more requests within timeout
            deadline = asyncio.get_running_loop().time() + BATCH_TIMEOUT_S
            while len(batch_data) < MAX_BATCH_SIZE:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch_data.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    return

            if batch_data:
                await self._run_batch(batch_data)

    async def _run_batch(self, batch_data: List[tuple]) -> None:
        """
        Process a batch of reranking requests.

        Args:
            batch_data: List of (query, documents, future) tuples
        """
        # Flatten all query-doc pairs
        all_pairs = []
        request_boundaries = []

        start_idx = 0
        for query, docs, _ in batch_data:
            pairs = [[query, doc] for doc in docs]
            all_pairs.extend(pairs)
            request_boundaries.append({
                "start": start_idx,
                "end": start_idx + len(pairs),
                "doc_count": len(docs),
            })
            start_idx += len(pairs)

        loop = asyncio.get_running_loop()

        try:
            t0 = time.perf_counter()

            # GPU inference in thread pool (FlagReranker is not async)
            scores = await loop.run_in_executor(
                None, lambda: reranker.compute_score(all_pairs)
            )

            latency_ms = (time.perf_counter() - t0) * 1000

            # Ensure scores is a list
            if not isinstance(scores, list):
                scores = [scores]

            # Redistribute results to original requests
            for i, (query, docs, future) in enumerate(batch_data):
                bounds = request_boundaries[i]
                doc_scores = scores[bounds["start"]:bounds["end"]]

                # Return (index, score) pairs sorted by score desc
                indexed_scores = list(enumerate(doc_scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)

                if not future.done():
                    future.set_result({
                        "results": indexed_scores,
                        "latency_ms": latency_ms,
                        "batch_size": len(all_pairs),
                    })

        except Exception as e:
            print(f"Rerank batch error: {e}")
            for _, _, future in batch_data:
                if not future.done():
                    future.set_exception(e)


batcher = RerankBatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global reranker

    print("Loading reranker model on GPU...")
    print("NOTE: This requires CUDA to be available on your system.")
    print(f"📦 Downloading/loading from: {MODELS_CACHE_DIR}")

    reranker = FlagReranker(
        MODEL_ID,
        use_fp16=True,  # Critical for 4GB VRAM (RTX 3050)
        device="cuda",
        cache_dir=str(MODELS_CACHE_DIR)
    )
    print(f"✅ Reranker loaded: {MODEL_ID} (FP16, CUDA)")
    print(f"✅ Models cached in: {MODELS_CACHE_DIR}")

    await batcher.start()

    yield

    await batcher.stop()
    print("Rerank API shutdown complete.")


app = FastAPI(
    title="BGE Reranker GPU API",
    description="Cross-encoder reranking with FP16 on RTX 3050",
    lifespan=lifespan,
)


class RerankRequest(BaseModel):
    """Reranking request model."""

    query: str
    documents: List[str]
    top_k: int = 10  # Return top K results


class RerankResult(BaseModel):
    """Single reranking result."""

    index: int
    score: float
    document: str


class RerankResponse(BaseModel):
    """Reranking response model."""

    results: List[RerankResult]
    latency_ms: float
    batch_size: int


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents for a query.

    Args:
        request: RerankRequest with query, documents, and top_k

    Returns:
        RerankResponse with ranked results
    """
    if not request.documents:
        raise HTTPException(400, "Documents list cannot be empty")

    if len(request.documents) > 100:
        raise HTTPException(400, "Maximum 100 documents per request")

    result = await batcher.process(request.query, request.documents)

    # Apply top_k and include document text
    top_results = []
    for idx, score in result["results"][:request.top_k]:
        top_results.append(
            RerankResult(
                index=idx,
                score=score,
                document=request.documents[idx],
            )
        )

    return RerankResponse(
        results=top_results,
        latency_ms=result["latency_ms"],
        batch_size=result["batch_size"],
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": "cuda",
        "precision": "fp16",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
