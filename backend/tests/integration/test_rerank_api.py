"""Integration tests for reranking API (via unified ML API)."""
import pytest
import httpx


# Unified ML API URL (rerank endpoint shares port with embed)
ML_API_URL = "http://localhost:8001"


@pytest.mark.integration
@pytest.mark.asyncio
class TestRerankAPI:
    """Test reranking API integration."""

    async def test_health_endpoint(self) -> None:
        """Test unified ML API health endpoint includes reranker status."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{ML_API_URL}/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                # Unified API returns nested model info
                assert "models" in data
                assert data["models"]["reranker"]["device"] == "cuda"
                assert data["models"]["reranker"]["precision"] == "fp16"
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_rerank_documents(self) -> None:
        """Test reranking functionality."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{ML_API_URL}/rerank",
                    json={
                        "query": "What is machine learning?",
                        "documents": [
                            "Machine learning is a subset of artificial intelligence.",
                            "The weather is nice today.",
                            "Neural networks are used in machine learning.",
                            "I like pizza for dinner.",
                        ],
                        "top_k": 2,
                    },
                )
                assert response.status_code == 200
                data = response.json()

                assert "results" in data
                assert len(data["results"]) == 2

                # First result should be most relevant (about ML)
                first_doc = data["results"][0]["document"].lower()
                assert "machine learning" in first_doc or "neural network" in first_doc

                # Scores should be in descending order
                assert data["results"][0]["score"] >= data["results"][1]["score"]

                # Should have latency info
                assert data["latency_ms"] > 0
                assert data["batch_size"] >= 4
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_rerank_returns_correct_indices(self) -> None:
        """Test that reranking returns correct document indices."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                docs = ["First doc", "Second doc", "Third doc"]
                response = await client.post(
                    f"{ML_API_URL}/rerank",
                    json={
                        "query": "second",
                        "documents": docs,
                        "top_k": 3,
                    },
                )
                assert response.status_code == 200
                data = response.json()

                # Verify indices are valid
                for result in data["results"]:
                    assert 0 <= result["index"] < len(docs)
                    assert result["document"] == docs[result["index"]]
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_rerank_empty_documents_error(self) -> None:
        """Test that empty documents list returns error."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{ML_API_URL}/rerank",
                    json={
                        "query": "test",
                        "documents": [],
                        "top_k": 5,
                    },
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_rerank_too_many_documents_error(self) -> None:
        """Test that >100 documents returns error."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                docs = [f"Document {i}" for i in range(101)]
                response = await client.post(
                    f"{ML_API_URL}/rerank",
                    json={
                        "query": "test",
                        "documents": docs,
                        "top_k": 10,
                    },
                )
                assert response.status_code == 400
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")

    async def test_rerank_handles_negative_scores(self) -> None:
        """Test that reranking handles negative cross-encoder scores."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # Use completely irrelevant documents to get negative scores
                response = await client.post(
                    f"{ML_API_URL}/rerank",
                    json={
                        "query": "quantum physics",
                        "documents": [
                            "xyz abc def ghi",
                            "123 456 789",
                            "random words here",
                        ],
                        "top_k": 3,
                    },
                )
                assert response.status_code == 200
                data = response.json()

                # Should still return results even with negative scores
                assert len(data["results"]) == 3
                # Scores can be negative for irrelevant docs
                # Just verify they're sorted correctly
                scores = [r["score"] for r in data["results"]]
                assert scores == sorted(scores, reverse=True)
            except httpx.ConnectError:
                pytest.skip("ML API not running on localhost:8001")
