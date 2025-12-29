"""Unit tests for Qdrant client."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.database.qdrant_client import QdrantDB


@pytest.mark.unit
class TestQdrantClient:
    """Test Qdrant client operations."""

    @patch("app.database.qdrant_client.QdrantClient")
    def test_init_collections(self, mock_qdrant: Mock) -> None:
        """Test collection initialization."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        # Should create both collections
        assert mock_client.create_collection.call_count == 2

    @patch("app.database.qdrant_client.QdrantClient")
    def test_collections_not_recreated_if_exist(self, mock_qdrant: Mock) -> None:
        """Test that existing collections are not recreated."""
        mock_client = Mock()
        mock_collection1 = Mock()
        mock_collection1.name = "chunks"
        mock_collection2 = Mock()
        mock_collection2.name = "parents"
        mock_client.get_collections.return_value.collections = [
            mock_collection1,
            mock_collection2,
        ]
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        # Should not create any collections
        mock_client.create_collection.assert_not_called()

    @patch("app.database.qdrant_client.QdrantClient")
    def test_file_exists_true(self, mock_qdrant: Mock) -> None:
        """Test file_exists returns True when file is indexed."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([Mock()], None)
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        result = db.file_exists("test-hash")
        assert result is True

    @patch("app.database.qdrant_client.QdrantClient")
    def test_file_exists_false(self, mock_qdrant: Mock) -> None:
        """Test file_exists returns False when file not indexed."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([], None)
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        result = db.file_exists("test-hash")
        assert result is False

    @patch("app.database.qdrant_client.QdrantClient")
    def test_store_chunks(self, mock_qdrant: Mock) -> None:
        """Test storing chunks with vectors."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        chunks = [
            {
                "id": "child-001",
                "content": "Test content",
                "parent_id": "parent-001",
                "file_hash": "hash123",
                "file_name": "test.pdf",
                "chunk_index": 0,
                "header_path": "Chapter 1",
            }
        ]
        dense_vectors = [[0.1] * 1024]  # 1024-dim vector
        sparse_vectors = [{1: 0.5, 10: 0.3}]

        db.store_chunks(chunks, dense_vectors, sparse_vectors)

        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "chunks"
        assert len(call_args[1]["points"]) == 1

    @patch("app.database.qdrant_client.QdrantClient")
    def test_store_parents(self, mock_qdrant: Mock) -> None:
        """Test storing parent chunks."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        parents = [
            {
                "id": "parent-001",
                "content": "Parent content",
                "file_hash": "hash123",
                "file_name": "test.pdf",
                "header_path": "Chapter 1",
                "child_ids": ["child-001", "child-002"],
            }
        ]

        db.store_parents(parents)

        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "parents"
        assert len(call_args[1]["points"]) == 1

    @patch("app.database.qdrant_client.QdrantClient")
    def test_hybrid_search(self, mock_qdrant: Mock) -> None:
        """Test hybrid search with RRF fusion."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_point = Mock()
        mock_point.id = "child-001"
        mock_point.score = 0.9
        mock_point.payload = {
            "content": "Test content",
            "parent_id": "parent-001",
            "file_name": "test.pdf",
            "header_path": "Chapter 1",
        }
        mock_result = Mock()
        mock_result.points = [mock_point]
        mock_client.query_points.return_value = mock_result
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        results = db.hybrid_search(
            query_dense=[0.1] * 1024,
            query_sparse={1: 0.5, 10: 0.3},
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["id"] == "child-001"
        assert results[0]["score"] == 0.9
        assert results[0]["content"] == "Test content"
        assert results[0]["parent_id"] == "parent-001"
        mock_client.query_points.assert_called_once()

    @patch("app.database.qdrant_client.QdrantClient")
    def test_hybrid_search_with_file_filter(self, mock_qdrant: Mock) -> None:
        """Test hybrid search with file hash filtering."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_result = Mock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        db.hybrid_search(
            query_dense=[0.1] * 1024,
            query_sparse={1: 0.5},
            file_hashes=["hash1", "hash2"],
            limit=10,
        )

        # Verify that query_points was called with filter
        call_args = mock_client.query_points.call_args
        assert call_args[1]["prefetch"] is not None
        # Both prefetch queries should have the filter
        for prefetch in call_args[1]["prefetch"]:
            assert prefetch.filter is not None

    @patch("app.database.qdrant_client.QdrantClient")
    def test_get_parents(self, mock_qdrant: Mock) -> None:
        """Test fetching parent chunks by IDs."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_point = Mock()
        mock_point.id = "parent-001"
        mock_point.payload = {
            "content": "Parent content",
            "file_name": "test.pdf",
            "header_path": "Chapter 1",
        }
        mock_client.retrieve.return_value = [mock_point]
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        parents = db.get_parents(["parent-001"])

        assert len(parents) == 1
        assert parents[0]["id"] == "parent-001"
        assert parents[0]["content"] == "Parent content"
        mock_client.retrieve.assert_called_once()

    @patch("app.database.qdrant_client.QdrantClient")
    def test_get_all_files(self, mock_qdrant: Mock) -> None:
        """Test getting list of all indexed files."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_point1 = Mock()
        mock_point1.payload = {"file_hash": "hash1", "file_name": "file1.pdf"}
        mock_point2 = Mock()
        mock_point2.payload = {"file_hash": "hash1", "file_name": "file1.pdf"}  # Duplicate
        mock_point3 = Mock()
        mock_point3.payload = {"file_hash": "hash2", "file_name": "file2.pdf"}

        # First call returns points, second call returns None (end of scroll)
        mock_client.scroll.side_effect = [
            ([mock_point1, mock_point2, mock_point3], None),
        ]
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        files = db.get_all_files()

        # Should deduplicate by file_hash
        assert len(files) == 2
        assert {"file_hash": "hash1", "file_name": "file1.pdf"} in files
        assert {"file_hash": "hash2", "file_name": "file2.pdf"} in files

    @patch("app.database.qdrant_client.QdrantClient")
    def test_delete_file(self, mock_qdrant: Mock) -> None:
        """Test deleting a file and all its chunks."""
        mock_client = Mock()
        mock_client.get_collections.return_value.collections = []
        mock_qdrant.return_value = mock_client

        db = QdrantDB()

        db.delete_file("hash123")

        # Should delete from both collections
        assert mock_client.delete.call_count == 2
        # Verify both calls were made with correct collection names
        call_args_list = mock_client.delete.call_args_list
        collections = [call[1]["collection_name"] for call in call_args_list]
        assert "chunks" in collections
        assert "parents" in collections
