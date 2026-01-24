"""Integration tests for embedding API."""

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
class TestEmbedAPI:
    """Test embedding API integration."""

    async def test_health_endpoint(self) -> None:
        """Test unified ML API health endpoint includes embedding status."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get("http://localhost:8001/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                # Unified API returns nested model info
                assert "models" in data
                assert data["models"]["embedding"]["device"] == "cuda"
                assert data["models"]["embedding"]["precision"] == "int8"
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_embed_single_text(self) -> None:
        """Test embedding single text."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": "Hello world", "is_query": False},
                )
                assert response.status_code == 200
                data = response.json()

                assert "dense_vecs" in data
                assert "sparse_vecs" in data
                assert len(data["dense_vecs"]) == 1
                assert len(data["dense_vecs"][0]) == 1024  # BGE-M3 dimension
                assert isinstance(data["sparse_vecs"][0], dict)
                assert data["latency_ms"] > 0
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_embed_multiple_texts(self) -> None:
        """Test embedding multiple texts in batch."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                texts = ["First document", "Second document", "Third document"]
                response = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": texts, "is_query": False},
                )
                assert response.status_code == 200
                data = response.json()

                assert len(data["dense_vecs"]) == 3
                assert len(data["sparse_vecs"]) == 3
                # Each vector should have correct dimension
                for vec in data["dense_vecs"]:
                    assert len(vec) == 1024
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_embed_with_query_prefix(self) -> None:
        """Test embedding with query prefix applied."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Embed same text with and without query prefix
                text = "machine learning"

                response_doc = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": text, "is_query": False},
                )
                response_query = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": text, "is_query": True},
                )

                assert response_doc.status_code == 200
                assert response_query.status_code == 200

                # Vectors should be different due to prefix
                vec_doc = response_doc.json()["dense_vecs"][0]
                vec_query = response_query.json()["dense_vecs"][0]

                # Should not be identical (query prefix changes embedding)
                assert vec_doc != vec_query
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_embed_empty_text_error(self) -> None:
        """Test that empty text returns error."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": [], "is_query": False},
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_embed_too_many_texts_error(self) -> None:
        """Test that exceeding MAX_TEXTS_PER_REQUEST returns error."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Create 150 texts (exceeds MAX_TEXTS_PER_REQUEST of 128)
                texts = [f"Text number {i}" for i in range(150)]
                response = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": texts, "is_query": False},
                )
                assert response.status_code == 400
                data = response.json()
                assert "Too many texts" in data["detail"]
                assert "128" in data["detail"]
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_embed_at_limit_succeeds(self) -> None:
        """Test that exactly MAX_TEXTS_PER_REQUEST texts succeeds."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Create exactly 128 texts (at MAX_TEXTS_PER_REQUEST limit)
                texts = [f"Text number {i}" for i in range(128)]
                response = await client.post(
                    "http://localhost:8001/embed",
                    json={"text": texts, "is_query": False},
                )
                assert response.status_code == 200
                data = response.json()
                assert len(data["dense_vecs"]) == 128
                assert len(data["sparse_vecs"]) == 128
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")
