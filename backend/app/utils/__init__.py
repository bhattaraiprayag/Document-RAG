"""Utility modules for the backend application."""
from .batch_embed import embed_texts_in_batches, DEFAULT_EMBED_BATCH_SIZE

__all__ = ["embed_texts_in_batches", "DEFAULT_EMBED_BATCH_SIZE"]
