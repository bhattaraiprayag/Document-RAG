"""
Unified ML API for Embedding and Reranking with GPU Acceleration.

This API provides:
- /embed: BGE-M3 ONNX INT8 embeddings (dense + sparse) on GPU
- /rerank: BGE-Reranker-Base FP16 reranking on GPU

Both models are loaded at startup for "hot" API performance.
Single port (8001) with path-based routing.
"""

import asyncio
import gc
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# CRITICAL: Set cache directories BEFORE any HuggingFace imports
MODELS_CACHE_DIR = Path("/models_cache")
MODELS_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(MODELS_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_CACHE_DIR / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_CACHE_DIR / "hub")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print(f"🔧 HF_HOME set to: {MODELS_CACHE_DIR}")

# NOW import HuggingFace libraries (after env vars are set)
from FlagEmbedding import FlagReranker  # noqa: E402
from optimum.onnxruntime import ORTModelForCustomTasks  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Embedding Model Config
EMBED_MODEL_ID = "BAAI/bge-m3"
EMBED_ONNX_MODEL_ID = "gpahal/bge-m3-onnx-int8"
EMBED_MAX_BATCH_SIZE = int(os.getenv("EMBED_MAX_BATCH_SIZE", "32"))
EMBED_BATCH_TIMEOUT_S = 0.01
EMBED_MAX_QUEUE_SIZE = 500
EMBED_MAX_TEXTS_PER_REQUEST = 128
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Rerank Model Config
RERANK_MODEL_ID = "BAAI/bge-reranker-base"
RERANK_MAX_BATCH_SIZE = int(os.getenv("RERANK_MAX_BATCH_SIZE", "16"))
RERANK_BATCH_TIMEOUT_S = float(os.getenv("RERANK_BATCH_TIMEOUT", "0.02"))
RERANK_MAX_QUEUE_SIZE = 100


model_resources = {
    "embed_tokenizer": None,
    "embed_model": None,
    "reranker": None,
}


class EmbedBatcher:
    """Async batching for embedding requests."""

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=EMBED_MAX_QUEUE_SIZE)
        self.processing_loop_task = None

    async def start(self):
        """Start the background processing loop."""
        self.processing_loop_task = asyncio.create_task(self._process_loop())
        print("✅ Embed batch processor started.")

    async def stop(self):
        """Stop the loop."""
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass

    async def process(self, texts: List[str]) -> dict:
        """Add work to the queue and await result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((texts, future))
        return await future

    async def _process_loop(self):
        """Collect requests and run inference in batches."""
        while True:
            batch_data = []

            try:
                item = await self.queue.get()
                batch_data.append(item)
            except asyncio.CancelledError:
                break

            deadline = asyncio.get_running_loop().time() + EMBED_BATCH_TIMEOUT_S
            while len(batch_data) < EMBED_MAX_BATCH_SIZE:
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

    async def _run_batch(self, batch_data: List[Tuple[List[str], asyncio.Future]]):
        """Flatten batch, run inference, redistribute results."""
        all_texts = []
        request_indices = []

        start_idx = 0
        for texts, _ in batch_data:
            all_texts.extend(texts)
            count = len(texts)
            request_indices.append((start_idx, start_idx + count))
            start_idx += count

        loop = asyncio.get_running_loop()

        try:
            dense_all, sparse_all, latency = await loop.run_in_executor(
                None, run_embed_inference_sync, all_texts
            )

            for i, (_, future) in enumerate(batch_data):
                start, end = request_indices[i]
                response_obj = {
                    "dense_vecs": dense_all[start:end],
                    "sparse_vecs": sparse_all[start:end],
                    "latency_ms": latency,
                    "batch_size": len(all_texts),
                }
                if not future.done():
                    future.set_result(response_obj)

        except Exception as e:
            print(f"❌ Embed Batch Error: {e}")
            for _, future in batch_data:
                if not future.done():
                    future.set_exception(e)


def run_embed_inference_sync(
    texts: List[str],
) -> Tuple[List[List[float]], List[Dict[int, float]], float]:
    """
    Synchronous embedding inference for thread pool execution.

    Uses ONNX Runtime GPU provider for INT8 quantized model.
    """
    tokenizer = model_resources["embed_tokenizer"]
    model = model_resources["embed_model"]

    if tokenizer is None or model is None:
        raise RuntimeError("Embedding model or tokenizer not initialized")

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",  # PyTorch tensors required for ONNX IO Binding
    )

    t0 = time.perf_counter()
    outputs = model(**inputs)
    latency = (time.perf_counter() - t0) * 1e3

    # Dense vectors
    dense_vecs = outputs["dense_vecs"].tolist()

    # Sparse vectors - remove special tokens
    sparse_vecs = []
    raw_sparse = outputs["sparse_vecs"]
    input_ids = inputs["input_ids"]

    for i, seq_weights in enumerate(raw_sparse):
        current_input_ids = input_ids[i]
        token_weight_map = {}
        for idx, weight in enumerate(seq_weights):
            if weight > 0:
                token_id = int(current_input_ids[idx])
                # Skip special tokens (BOS=0, PAD=1, EOS=2)
                if token_id in [0, 1, 2]:
                    continue
                val = weight.item()
                if token_id in token_weight_map:
                    token_weight_map[token_id] = max(token_weight_map[token_id], val)
                else:
                    token_weight_map[token_id] = val
        sparse_vecs.append(token_weight_map)

    # Cleanup
    del inputs
    del outputs
    del raw_sparse
    del input_ids
    gc.collect()

    return dense_vecs, sparse_vecs, latency


class RerankBatcher:
    """Async batching for reranking requests."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=RERANK_MAX_QUEUE_SIZE)
        self.processing_loop_task = None

    async def start(self):
        """Start the background processing loop."""
        self.processing_loop_task = asyncio.create_task(self._process_loop())
        print("✅ Rerank batch processor started.")

    async def stop(self):
        """Stop the processing loop."""
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass

    async def process(self, query: str, documents: List[str]) -> dict:
        """Process a reranking request."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((query, documents, future))
        return await future

    async def _process_loop(self):
        """Background loop that collects and processes batches."""
        while True:
            batch_data = []

            try:
                item = await self.queue.get()
                batch_data.append(item)
            except asyncio.CancelledError:
                break

            deadline = asyncio.get_running_loop().time() + RERANK_BATCH_TIMEOUT_S
            while len(batch_data) < RERANK_MAX_BATCH_SIZE:
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

    async def _run_batch(self, batch_data: List[tuple]):
        """Process a batch of reranking requests."""
        all_pairs = []
        request_boundaries = []

        start_idx = 0
        for query, docs, _ in batch_data:
            pairs = [[query, doc] for doc in docs]
            all_pairs.extend(pairs)
            request_boundaries.append(
                {
                    "start": start_idx,
                    "end": start_idx + len(pairs),
                    "doc_count": len(docs),
                }
            )
            start_idx += len(pairs)

        loop = asyncio.get_running_loop()

        try:
            t0 = time.perf_counter()

            reranker = model_resources["reranker"]
            if reranker is None:
                raise RuntimeError("Reranker model not initialized")
            scores = await loop.run_in_executor(
                None, lambda: reranker.compute_score(all_pairs)
            )

            latency_ms = (time.perf_counter() - t0) * 1000

            if not isinstance(scores, list):
                scores = [scores]

            for i, (_, _, future) in enumerate(batch_data):
                bounds = request_boundaries[i]
                doc_scores = scores[bounds["start"] : bounds["end"]]

                indexed_scores = list(enumerate(doc_scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)

                if not future.done():
                    future.set_result(
                        {
                            "results": indexed_scores,
                            "latency_ms": latency_ms,
                            "batch_size": len(all_pairs),
                        }
                    )

        except Exception as e:
            print(f"❌ Rerank batch error: {e}")
            for _, _, future in batch_data:
                if not future.done():
                    future.set_exception(e)


embed_batcher = EmbedBatcher()
rerank_batcher = RerankBatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load both models at startup."""
    print("=" * 60)
    print("🚀 Starting Unified ML API (GPU)")
    print("=" * 60)

    # Load Embedding Model (ONNX INT8 on GPU)
    print("\n📦 Loading BGE-M3 Embedding Model (ONNX INT8, GPU)...")
    model_resources["embed_tokenizer"] = AutoTokenizer.from_pretrained(
        EMBED_MODEL_ID, cache_dir=str(MODELS_CACHE_DIR)
    )

    # Use ONNX Runtime with CUDA Execution Provider
    # use_io_binding=False: Required because this model has dynamic axes
    # that IO Binding cannot handle (causes "invalid literal for int()" error)
    model_resources["embed_model"] = ORTModelForCustomTasks.from_pretrained(
        EMBED_ONNX_MODEL_ID,
        file_name="model_quantized.onnx",
        cache_dir=str(MODELS_CACHE_DIR),
        provider="CUDAExecutionProvider",
        use_io_binding=False,
    )
    print(f"✅ Embedding model loaded: {EMBED_ONNX_MODEL_ID}")

    # Load Reranker Model (FP16 on GPU)
    print("\n📦 Loading BGE-Reranker Model (FP16, GPU)...")
    model_resources["reranker"] = FlagReranker(
        RERANK_MODEL_ID, use_fp16=True, device="cuda", cache_dir=str(MODELS_CACHE_DIR)
    )
    print(f"✅ Reranker model loaded: {RERANK_MODEL_ID}")

    # Start batchers
    print("\n🔄 Starting batch processors...")
    await embed_batcher.start()
    await rerank_batcher.start()

    print("\n" + "=" * 60)
    print("✅ ML API Ready - Both models loaded on GPU")
    print("=" * 60 + "\n")

    yield

    # Cleanup
    await embed_batcher.stop()
    await rerank_batcher.stop()
    model_resources.clear()
    print("🛑 ML API shutdown complete.")


app = FastAPI(
    title="Unified ML API",
    description="GPU-accelerated Embedding and Reranking API",
    version="1.0.0",
    lifespan=lifespan,
)


class EmbeddingRequest(BaseModel):
    """Embedding request model."""

    text: Union[str, List[str]]
    is_query: bool = False


class EmbeddingResponse(BaseModel):
    """Embedding response model."""

    dense_vecs: List[List[float]]
    sparse_vecs: List[Dict[int, float]]
    latency_ms: float
    batch_size: int


class RerankRequest(BaseModel):
    """Reranking request model."""

    query: str
    documents: List[str]
    top_k: int = 10


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


@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for text(s).

    Args:
        request: EmbeddingRequest with text and optional is_query flag

    Returns:
        EmbeddingResponse with dense and sparse vectors
    """
    input_texts = [request.text] if isinstance(request.text, str) else request.text

    if not input_texts:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if len(input_texts) > EMBED_MAX_TEXTS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many texts. Max: {EMBED_MAX_TEXTS_PER_REQUEST}, "
                f"got: {len(input_texts)}"
            ),
        )

    if request.is_query:
        input_texts = [QUERY_PREFIX + t for t in input_texts]

    result = await embed_batcher.process(input_texts)

    return EmbeddingResponse(
        dense_vecs=result["dense_vecs"],
        sparse_vecs=result["sparse_vecs"],
        latency_ms=result["latency_ms"],
        batch_size=result["batch_size"],
    )


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

    result = await rerank_batcher.process(request.query, request.documents)

    top_results = []
    for idx, score in result["results"][: request.top_k]:
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
    """Combined health check for both models."""
    embed_ready = model_resources.get("embed_model") is not None
    rerank_ready = model_resources.get("reranker") is not None

    if not embed_ready or not rerank_ready:
        raise HTTPException(503, "Models not loaded")

    return {
        "status": "healthy",
        "models": {
            "embedding": {
                "model": EMBED_ONNX_MODEL_ID,
                "device": "cuda",
                "precision": "int8",
                "ready": embed_ready,
            },
            "reranker": {
                "model": RERANK_MODEL_ID,
                "device": "cuda",
                "precision": "fp16",
                "ready": rerank_ready,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
