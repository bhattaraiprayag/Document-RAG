"""
BGE-M3 ONNX Embedding API (CPU).
Produces dense (1024-dim) + sparse vectors for hybrid search.

Enhanced version with query prefix support for better retrieval.
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
import gc
import time
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union, Tuple
from optimum.onnxruntime import ORTModelForCustomTasks
from transformers import AutoTokenizer


# --- Configuration ---
MODEL_ID = "BAAI/bge-m3"
ONNX_MODEL_ID = "gpahal/bge-m3-onnx-int8"
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
BATCH_TIMEOUT_S = 0.01
MAX_QUEUE_SIZE = 500  # Reduced from 1000 for better backpressure
MAX_TEXTS_PER_REQUEST = 128  # Hard limit to prevent memory explosion

# Query instruction prefix for better retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# --- Global Resources ---
model_resources = {}


# --- The Dynamic Batcher ---
class ModelBatcher:
    """Async batching for embedding requests."""

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.processing_loop_task = None

    async def start(self):
        """Starts the background processing loop."""
        self.processing_loop_task = asyncio.create_task(self._process_loop())
        print("Batch processing loop started.")

    async def stop(self):
        """Stops the loop nicely."""
        if self.processing_loop_task:
            self.processing_loop_task.cancel()
            try:
                await self.processing_loop_task
            except asyncio.CancelledError:
                pass

    async def process(self, texts: List[str]):
        """
        Public method for API to add work.
        Returns a Future that will resolve with the result.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self.queue.put((texts, future))

        return await future

    async def _process_loop(self):
        """
        The heartbeat: collects requests and runs inference in batches.
        """
        while True:
            batch_data = []

            # 1. Fetch first item (blocking wait)
            try:
                item = await self.queue.get()
                batch_data.append(item)
            except asyncio.CancelledError:
                break

            # 2. Try to fill the batch with whatever is immediately available
            # or wait a tiny bit (BATCH_TIMEOUT_S) to accumulate more.
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

            # 3. Process the batch
            if batch_data:
                await self._run_batch(batch_data)

    async def _run_batch(self, batch_data: List[Tuple[List[str], asyncio.Future]]):
        """
        Flattens the batch, runs inference, and redistributes results.
        """
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
                None, run_inference_sync, all_texts
            )

            for i, (original_texts, future) in enumerate(batch_data):
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
            print(f"Batch Inference Error: {e}")
            for _, future in batch_data:
                if not future.done():
                    future.set_exception(e)


# --- Helper Function for Inference (Synchronous) ---
def run_inference_sync(texts: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]], float]:
    """
    Synchronous inference function for thread pool execution.

    Includes explicit memory cleanup to prevent accumulation across batches.

    Args:
        texts: List of text strings to embed

    Returns:
        Tuple of (dense_vecs, sparse_vecs, latency_ms)
    """
    tokenizer = model_resources["tokenizer"]
    model = model_resources["model"]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    t0 = time.perf_counter()
    outputs = model(**inputs)
    latency = (time.perf_counter() - t0) * 1e3

    # Dense vectors - convert to Python lists immediately
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

    # Explicit memory cleanup to prevent accumulation
    del inputs
    del outputs
    del raw_sparse
    del input_ids
    gc.collect()

    return dense_vecs, sparse_vecs, latency


# --- FastAPI Setup ---
batcher = ModelBatcher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("Loading BGE-M3 ONNX model (CPU)...")
    print(f"📦 Downloading/loading from: {MODELS_CACHE_DIR}")

    model_resources["tokenizer"] = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODELS_CACHE_DIR)
    )
    model_resources["model"] = ORTModelForCustomTasks.from_pretrained(
        ONNX_MODEL_ID,
        file_name="model_quantized.onnx",
        cache_dir=str(MODELS_CACHE_DIR)
    )
    print(f"✅ Model loaded: {ONNX_MODEL_ID} (ONNX INT8, CPU)")
    print(f"✅ Models cached in: {MODELS_CACHE_DIR}")

    # Start the batcher
    await batcher.start()

    yield

    # Cleanup
    await batcher.stop()
    model_resources.clear()
    print("Embedding API shutdown complete.")


app = FastAPI(
    title="BGE-M3 ONNX Embedding API",
    description="Dense + Sparse embeddings for hybrid search (CPU)",
    lifespan=lifespan,
)


class EmbeddingRequest(BaseModel):
    """Embedding request model."""

    text: Union[str, List[str]]
    is_query: bool = False  # Set True to apply query prefix


class EmbeddingResponse(BaseModel):
    """Embedding response model."""

    dense_vecs: List[List[float]]
    sparse_vecs: List[Dict[int, float]]
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

    Raises:
        HTTPException 400: If input is empty or exceeds MAX_TEXTS_PER_REQUEST
    """
    input_texts = [request.text] if isinstance(request.text, str) else request.text

    if not input_texts:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if len(input_texts) > MAX_TEXTS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts in single request. Max: {MAX_TEXTS_PER_REQUEST}, got: {len(input_texts)}. "
            f"Use smaller batches to prevent memory issues.",
        )

    # Apply query prefix if flagged
    if request.is_query:
        input_texts = [QUERY_PREFIX + t for t in input_texts]

    result = await batcher.process(input_texts)

    return EmbeddingResponse(
        dense_vecs=result["dense_vecs"],
        sparse_vecs=result["sparse_vecs"],
        latency_ms=result["latency_ms"],
        batch_size=result["batch_size"],
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": ONNX_MODEL_ID,
        "device": "cpu",
        "precision": "int8",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
