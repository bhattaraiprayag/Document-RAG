"""Utility modules for the backend application."""

from .batch_embed import DEFAULT_EMBED_BATCH_SIZE, embed_texts_in_batches

__all__ = ["embed_texts_in_batches", "DEFAULT_EMBED_BATCH_SIZE"]
