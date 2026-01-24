"""Unit tests for RAG orchestrator."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.config import settings
from app.rag.orchestrator import RAGOrchestrator, RetrievedContext


@pytest.mark.unit
class TestRAGOrchestrator:
    """Test RAG orchestration logic."""

    @pytest.mark.asyncio
    @patch("app.rag.orchestrator.QdrantDB")
    @patch("app.rag.orchestrator.ModelFactory")
    async def test_embed_query(
        self, mock_factory_class: Mock, mock_db_class: Mock
    ) -> None:
        """Test query embedding."""
        # Mock HTTP client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={
                "dense_vecs": [[0.1, 0.2, 0.3]],
                "sparse_vecs": [{1: 0.5, 10: 0.3}],
            }
        )
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)

        orchestrator = RAGOrchestrator()
        orchestrator.http_client = mock_client

        result = await orchestrator._embed_query("test query")

        assert "dense" in result
        assert "sparse" in result
        assert result["dense"] == [0.1, 0.2, 0.3]
        assert result["sparse"] == {1: 0.5, 10: 0.3}
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.rag.orchestrator.QdrantDB")
    @patch("app.rag.orchestrator.ModelFactory")
    async def test_rerank(self, mock_factory_class: Mock, mock_db_class: Mock) -> None:
        """Test document reranking."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={
                "results": [
                    {"index": 1, "score": 0.9},
                    {"index": 0, "score": 0.7},
                ],
            }
        )
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)

        orchestrator = RAGOrchestrator()
        orchestrator.http_client = mock_client

        result = await orchestrator._rerank("query", ["doc1", "doc2"])

        assert len(result) == 2
        assert result[0] == (1, 0.9)  # Higher score first
        assert result[1] == (0, 0.7)

    @pytest.mark.asyncio
    @patch("app.rag.orchestrator.QdrantDB")
    @patch("app.rag.orchestrator.ModelFactory")
    async def test_query_no_results(
        self, mock_factory_class: Mock, mock_db_class: Mock
    ) -> None:
        """Test query when no results are found."""
        # Mock database to return empty results
        mock_db = Mock()
        mock_db.hybrid_search.return_value = []
        mock_db_class.return_value = mock_db

        # Mock embed query
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={
                "dense_vecs": [[0.1] * 1024],
                "sparse_vecs": [{1: 0.5}],
            }
        )
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)

        orchestrator = RAGOrchestrator()
        orchestrator.http_client = mock_client
        orchestrator.db = mock_db

        tokens = []
        async for token in orchestrator.query("test query"):
            tokens.append(token)

        response = "".join(tokens)
        assert "couldn't find" in response.lower() or "no relevant" in response.lower()

    @pytest.mark.asyncio
    @patch("app.rag.orchestrator.QdrantDB")
    @patch("app.rag.orchestrator.ModelFactory")
    async def test_query_with_file_filter(
        self, mock_factory_class: Mock, mock_db_class: Mock
    ) -> None:
        """Test query with file filtering."""
        mock_db = Mock()
        mock_db.hybrid_search.return_value = []
        mock_db_class.return_value = mock_db

        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json = Mock(
            return_value={
                "dense_vecs": [[0.1] * 1024],
                "sparse_vecs": [{1: 0.5}],
            }
        )
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)

        orchestrator = RAGOrchestrator()
        orchestrator.http_client = mock_client
        orchestrator.db = mock_db

        selected_files = ["hash1", "hash2"]
        async for _ in orchestrator.query("test", selected_files=selected_files):
            pass

        # Verify hybrid_search was called with file filter
        mock_db.hybrid_search.assert_called_once()
        call_args = mock_db.hybrid_search.call_args
        assert call_args[1]["file_hashes"] == selected_files

    @pytest.mark.asyncio
    @patch("app.rag.orchestrator.QdrantDB")
    @patch("app.rag.orchestrator.ModelFactory")
    async def test_parent_deduplication(
        self, mock_factory_class: Mock, mock_db_class: Mock
    ) -> None:
        """Test that duplicate parents are deduplicated."""
        # Setup mocks
        mock_db = Mock()
        mock_db.hybrid_search.return_value = [
            {
                "content": "child1",
                "parent_id": "parent-001",
                "file_name": "test.pdf",
                "header_path": "Ch1",
            },
            {
                "content": "child2",
                "parent_id": "parent-001",
                "file_name": "test.pdf",
                "header_path": "Ch1",
            },  # Same parent
            {
                "content": "child3",
                "parent_id": "parent-002",
                "file_name": "test.pdf",
                "header_path": "Ch2",
            },
        ]
        mock_db.get_parents.return_value = [
            {
                "id": "parent-001",
                "content": "Parent 1 content",
                "file_name": "test.pdf",
                "header_path": "Ch1",
            },
            {
                "id": "parent-002",
                "content": "Parent 2 content",
                "file_name": "test.pdf",
                "header_path": "Ch2",
            },
        ]
        mock_db_class.return_value = mock_db

        # Mock embed and rerank
        mock_client = AsyncMock()
        embed_response = Mock()
        embed_response.json = Mock(
            return_value={"dense_vecs": [[0.1] * 1024], "sparse_vecs": [{1: 0.5}]}
        )
        embed_response.raise_for_status = Mock()

        rerank_response = Mock()
        rerank_response.json = Mock(
            return_value={
                "results": [
                    {"index": 0, "score": 0.9},
                    {"index": 1, "score": 0.8},
                    {"index": 2, "score": 0.7},
                ]
            }
        )
        rerank_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(side_effect=[embed_response, rerank_response])

        # Mock LLM provider
        async def mock_generate(*args, **kwargs):
            yield "Response text"

        mock_provider = Mock()
        mock_provider.generate_streaming = mock_generate
        mock_factory = Mock()
        mock_factory.get_provider.return_value = mock_provider
        mock_factory_class.return_value = mock_factory

        orchestrator = RAGOrchestrator()
        orchestrator.http_client = mock_client
        orchestrator.db = mock_db
        orchestrator.llm_provider = mock_provider

        async for _ in orchestrator.query("test"):
            pass

        # Verify get_parents was called with deduplicated parent IDs
        mock_db.get_parents.assert_called_once()
        parent_ids = mock_db.get_parents.call_args[0][0]
        # Should only have 2 unique parents, not 3
        assert len(parent_ids) == 2
        assert "parent-001" in parent_ids
        assert "parent-002" in parent_ids


@pytest.mark.unit
class TestRetrievedContext:
    """Test RetrievedContext dataclass."""

    def test_retrieved_context_creation(self) -> None:
        """Test creating RetrievedContext object."""
        context = RetrievedContext(
            parent_id="parent-001",
            parent_content="Test content",
            file_name="test.pdf",
            header_path="Chapter 1",
            child_score=0.95,
        )

        assert context.parent_id == "parent-001"
        assert context.parent_content == "Test content"
        assert context.file_name == "test.pdf"
        assert context.header_path == "Chapter 1"
        assert context.child_score == 0.95


@pytest.mark.unit
class TestRAGOrchestratorConfiguration:
    """Test RAG orchestrator configuration."""

    @patch("app.rag.orchestrator.QdrantDB")
    @patch("app.rag.orchestrator.ModelFactory")
    def test_http_client_timeout_uses_settings(
        self, mock_factory_class: Mock, mock_db_class: Mock
    ) -> None:
        """Test that HTTP client uses configurable timeout from settings."""
        orchestrator = RAGOrchestrator()
        timeout = orchestrator.http_client.timeout
        assert timeout.connect == settings.rag_http_timeout
        assert timeout.read == settings.rag_http_timeout

    def test_default_timeout_is_120_seconds(self) -> None:
        """Test that default timeout is 120 seconds for rerank operations."""
        assert settings.rag_http_timeout == 120.0
