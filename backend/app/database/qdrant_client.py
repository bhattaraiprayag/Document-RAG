"""Qdrant vector database client."""
from typing import Any, Optional

from qdrant_client import QdrantClient, models

from ..config import settings

CHUNKS_COLLECTION = "chunks"
PARENTS_COLLECTION = "parents"
DENSE_VECTOR_SIZE = 1024
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantDB:
    """Qdrant database client for vector storage and retrieval."""

    def __init__(self) -> None:
        """Initialize Qdrant client and create collections if needed."""
        self.client = QdrantClient(url=settings.qdrant_url)
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialize chunks and parents collections."""
        collections = [c.name for c in self.client.get_collections().collections]

        # CHUNKS COLLECTION (retrieval targets with vectors)
        if CHUNKS_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=CHUNKS_COLLECTION,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )

        # PARENTS COLLECTION (context source, no vectors)
        if PARENTS_COLLECTION not in collections:
            self.client.create_collection(
                collection_name=PARENTS_COLLECTION, vectors_config={}
            )

    def file_exists(self, file_hash: str) -> bool:
        """
        Check if a file has already been indexed.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if file exists in database, False otherwise
        """
        result = self.client.scroll(
            collection_name=PARENTS_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash", match=models.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1,
        )
        return len(result[0]) > 0

    def store_chunks(
        self,
        chunks: list[dict[str, Any]],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> None:
        """
        Store child chunks with their vectors.

        Args:
            chunks: List of chunk metadata dicts
            dense_vectors: Dense embedding vectors (1024-dim)
            sparse_vectors: Sparse vectors as {index: value} dicts
        """
        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=chunk["id"],
                    vector={
                        DENSE_VECTOR_NAME: dense_vectors[i],
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=list(sparse_vectors[i].keys()),
                            values=list(sparse_vectors[i].values()),
                        ),
                    },
                    payload={
                        "content": chunk["content"],
                        "parent_id": chunk["parent_id"],
                        "file_hash": chunk["file_hash"],
                        "file_name": chunk["file_name"],
                        "chunk_index": chunk["chunk_index"],
                        "header_path": chunk["header_path"],
                    },
                )
            )

        self.client.upsert(collection_name=CHUNKS_COLLECTION, points=points)

    def store_parents(self, parents: list[dict[str, Any]]) -> None:
        """
        Store parent chunks (no vectors).

        Args:
            parents: List of parent chunk metadata dicts
        """
        points = []
        for parent in parents:
            points.append(
                models.PointStruct(
                    id=parent["id"],
                    vector={},
                    payload={
                        "content": parent["content"],
                        "file_hash": parent["file_hash"],
                        "file_name": parent["file_name"],
                        "header_path": parent["header_path"],
                        "child_ids": parent["child_ids"],
                    },
                )
            )

        self.client.upsert(collection_name=PARENTS_COLLECTION, points=points)

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: dict[int, float],
        file_hashes: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search with RRF fusion.

        Args:
            query_dense: Dense query vector
            query_sparse: Sparse query vector
            file_hashes: Optional list of file hashes to filter by
            limit: Maximum results to return

        Returns:
            List of child chunks with metadata
        """
        filter_conditions = None
        if file_hashes:
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash", match=models.MatchAny(any=file_hashes)
                    )
                ]
            )

        prefetch = [
            models.Prefetch(
                query=query_dense,
                using=DENSE_VECTOR_NAME,
                limit=limit * 2,
                filter=filter_conditions,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(query_sparse.keys()),
                    values=list(query_sparse.values()),
                ),
                using=SPARSE_VECTOR_NAME,
                limit=limit * 2,
                filter=filter_conditions,
            ),
        ]

        results = self.client.query_points(
            collection_name=CHUNKS_COLLECTION,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "content": point.payload["content"],  # type: ignore
                "parent_id": point.payload["parent_id"],  # type: ignore
                "file_name": point.payload["file_name"],  # type: ignore
                "header_path": point.payload["header_path"],  # type: ignore
            }
            for point in results.points
        ]

    def get_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """
        Fetch parent chunks by their IDs.

        Args:
            parent_ids: List of parent chunk IDs

        Returns:
            List of parent chunks with content
        """
        results = self.client.retrieve(
            collection_name=PARENTS_COLLECTION,
            ids=parent_ids,
            with_payload=True,
        )

        return [
            {
                "id": point.id,
                "content": point.payload["content"],  # type: ignore
                "file_name": point.payload["file_name"],  # type: ignore
                "header_path": point.payload["header_path"],  # type: ignore
            }
            for point in results
        ]

    def get_all_files(self) -> list[dict[str, str]]:
        """
        Get list of all indexed files.

        Returns:
            List of dicts with file_hash and file_name
        """
        files: dict[str, dict[str, str]] = {}
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=PARENTS_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=["file_hash", "file_name"],
            )

            for point in results:
                if point.payload is None:
                    continue
                file_hash = point.payload["file_hash"]
                if file_hash not in files:
                    files[file_hash] = {
                        "file_hash": file_hash,
                        "file_name": point.payload["file_name"],
                    }

            if offset is None:
                break

        return list(files.values())

    def delete_file(self, file_hash: str) -> None:
        """
        Delete all chunks and parents for a file.

        Args:
            file_hash: SHA256 hash of the file to delete
        """
        filter_selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_hash", match=models.MatchValue(value=file_hash)
                    )
                ]
            )
        )

        self.client.delete(
            collection_name=CHUNKS_COLLECTION, points_selector=filter_selector
        )

        self.client.delete(
            collection_name=PARENTS_COLLECTION, points_selector=filter_selector
        )
