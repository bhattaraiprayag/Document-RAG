"""Utility functions for batch embedding operations."""
import asyncio
from typing import Any, Optional

# Default batch size for embedding operations
DEFAULT_EMBED_BATCH_SIZE = 64


async def embed_texts_in_batches(
    client: Any,
    texts: list[str],
    embed_api_url: str,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    progress_callback: Optional[callable] = None,
) -> tuple[list[list[float]], list[dict[int, float]], float]:
    """
    Embed texts in controlled batches to prevent memory explosion.

    Instead of sending all texts in a single request (which can cause RAM
    to spike to 100% on large documents), this function splits the texts
    into smaller batches and processes them sequentially.

    Args:
        client: httpx.AsyncClient instance
        texts: List of text strings to embed
        embed_api_url: Base URL of the embedding API
        batch_size: Number of texts per batch (default 64)
        progress_callback: Optional callback(current_batch, total_batches) for progress

    Returns:
        Tuple of (dense_vectors, sparse_vectors, total_latency_ms)
    """
    if not texts:
        return [], [], 0.0

    all_dense: list[list[float]] = []
    all_sparse: list[dict[int, float]] = []
    total_latency = 0.0

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx : batch_idx + batch_size]
        current_batch_num = batch_idx // batch_size + 1

        response = await client.post(
            f"{embed_api_url}/embed",
            json={"text": batch_texts, "is_query": False},
        )
        response.raise_for_status()
        embed_data = response.json()

        all_dense.extend(embed_data["dense_vecs"])
        all_sparse.extend(embed_data["sparse_vecs"])
        total_latency += embed_data["latency_ms"]

        if progress_callback:
            progress_callback(current_batch_num, total_batches)

        # Yield control to event loop between batches
        await asyncio.sleep(0)

    return all_dense, all_sparse, total_latency
