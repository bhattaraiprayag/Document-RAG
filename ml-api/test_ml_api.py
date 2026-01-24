"""
Tests for the unified ML API.

Run with: pytest test_ml_api.py -v
"""
import httpx
import pytest

# Test configuration
ML_API_URL = "http://localhost:8001"


@pytest.fixture
def api_url() -> str:
    """Return the ML API URL."""
    return ML_API_URL


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, api_url: str):
        """Test that health endpoint returns 200 when API is running."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url}/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert "models" in data
                assert "embedding" in data["models"]
                assert "reranker" in data["models"]
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_health_shows_gpu_device(self, api_url: str):
        """Test that health endpoint shows GPU device for both models."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url}/health")
                data = response.json()
                assert data["models"]["embedding"]["device"] == "cuda"
                assert data["models"]["reranker"]["device"] == "cuda"
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")


class TestEmbedEndpoint:
    """Tests for the /embed endpoint."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, api_url: str):
        """Test embedding a single text string."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_url}/embed",
                    json={"text": "Hello, world!", "is_query": False},
                )
                assert response.status_code == 200
                data = response.json()
                assert "dense_vecs" in data
                assert "sparse_vecs" in data
                assert len(data["dense_vecs"]) == 1
                assert len(data["dense_vecs"][0]) == 1024  # BGE-M3 dimension
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, api_url: str):
        """Test embedding multiple texts."""
        async with httpx.AsyncClient() as client:
            try:
                texts = ["First text", "Second text", "Third text"]
                response = await client.post(
                    f"{api_url}/embed",
                    json={"text": texts, "is_query": False},
                )
                assert response.status_code == 200
                data = response.json()
                assert len(data["dense_vecs"]) == 3
                assert len(data["sparse_vecs"]) == 3
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_embed_with_query_prefix(self, api_url: str):
        """Test embedding with is_query=True applies query prefix."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_url}/embed",
                    json={"text": "What is machine learning?", "is_query": True},
                )
                assert response.status_code == 200
                data = response.json()
                assert len(data["dense_vecs"]) == 1
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_embed_empty_text_returns_400(self, api_url: str):
        """Test that empty text returns 400 error."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_url}/embed",
                    json={"text": [], "is_query": False},
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_embed_too_many_texts_returns_400(self, api_url: str):
        """Test that exceeding max texts returns 400 error."""
        async with httpx.AsyncClient() as client:
            try:
                texts = ["text"] * 150  # Exceeds MAX_TEXTS_PER_REQUEST (128)
                response = await client.post(
                    f"{api_url}/embed",
                    json={"text": texts, "is_query": False},
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")


class TestRerankEndpoint:
    """Tests for the /rerank endpoint."""

    @pytest.mark.asyncio
    async def test_rerank_basic(self, api_url: str):
        """Test basic reranking functionality."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_url}/rerank",
                    json={
                        "query": "What is Python?",
                        "documents": [
                            "Python is a programming language.",
                            "Java is a programming language.",
                            "The sun is a star.",
                        ],
                        "top_k": 2,
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 2
                # First result should be most relevant (Python doc)
                assert data["results"][0]["index"] == 0
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_rerank_returns_sorted_scores(self, api_url: str):
        """Test that rerank results are sorted by score descending."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_url}/rerank",
                    json={
                        "query": "machine learning",
                        "documents": [
                            "Cooking recipes for beginners.",
                            "Machine learning is a branch of AI.",
                            "Weather forecast for tomorrow.",
                        ],
                        "top_k": 3,
                    },
                )
                data = response.json()
                scores = [r["score"] for r in data["results"]]
                assert scores == sorted(scores, reverse=True)
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_rerank_empty_documents_returns_400(self, api_url: str):
        """Test that empty documents list returns 400 error."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_url}/rerank",
                    json={
                        "query": "test query",
                        "documents": [],
                        "top_k": 5,
                    },
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    @pytest.mark.asyncio
    async def test_rerank_too_many_documents_returns_400(self, api_url: str):
        """Test that exceeding max documents returns 400 error."""
        async with httpx.AsyncClient() as client:
            try:
                documents = ["doc"] * 101  # Exceeds max (100)
                response = await client.post(
                    f"{api_url}/rerank",
                    json={
                        "query": "test query",
                        "documents": documents,
                        "top_k": 5,
                    },
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
