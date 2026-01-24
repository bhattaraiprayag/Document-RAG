"""Unit tests for batch embedding functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.utils.batch_embed import DEFAULT_EMBED_BATCH_SIZE, embed_texts_in_batches


class TestEmbedTextsInBatches:
    """Test the embed_texts_in_batches function."""

    @pytest.mark.asyncio
    async def test_batch_streaming_splits_correctly(self) -> None:
        """Test that texts are split into correct batch sizes."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "dense_vecs": [[0.1] * 1024] * 64,
            "sparse_vecs": [{1: 0.5}] * 64,
            "latency_ms": 100.0,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        # 200 texts should result in 4 batches (64+64+64+8)
        texts = [f"Text {i}" for i in range(200)]

        dense, sparse, latency = await embed_texts_in_batches(
            client=mock_client,
            texts=texts,
            embed_api_url="http://localhost:8001",
            batch_size=64,
        )

        # Should have made 4 POST requests
        assert mock_client.post.call_count == 4

        # Check batch sizes in each call
        call_args_list = mock_client.post.call_args_list
        batch_sizes = [len(call.kwargs["json"]["text"]) for call in call_args_list]
        assert batch_sizes == [64, 64, 64, 8]

    @pytest.mark.asyncio
    async def test_batch_streaming_accumulates_results(self) -> None:
        """Test that results from all batches are accumulated correctly."""
        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            batch_size = len(kwargs["json"]["text"])

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "dense_vecs": [[float(call_count)] * 1024] * batch_size,
                "sparse_vecs": [{call_count: 0.5}] * batch_size,
                "latency_ms": 50.0,
            }
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client = AsyncMock()
        mock_client.post.side_effect = mock_post

        texts = [f"Text {i}" for i in range(150)]  # 3 batches: 64+64+22

        dense, sparse, latency = await embed_texts_in_batches(
            client=mock_client,
            texts=texts,
            embed_api_url="http://localhost:8001",
            batch_size=64,
        )

        # Should have all 150 results
        assert len(dense) == 150
        assert len(sparse) == 150

        # Total latency should be sum of all batches
        assert latency == 150.0  # 3 batches * 50ms

    @pytest.mark.asyncio
    async def test_batch_streaming_progress_callback(self) -> None:
        """Test that progress callback is called for each batch."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "dense_vecs": [[0.1] * 1024] * 32,
            "sparse_vecs": [{1: 0.5}] * 32,
            "latency_ms": 25.0,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        progress_calls = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        texts = [f"Text {i}" for i in range(100)]  # 4 batches with batch_size=32

        await embed_texts_in_batches(
            client=mock_client,
            texts=texts,
            embed_api_url="http://localhost:8001",
            batch_size=32,
            progress_callback=on_progress,
        )

        # Should have 4 progress calls
        assert len(progress_calls) == 4
        assert progress_calls == [(1, 4), (2, 4), (3, 4), (4, 4)]

    @pytest.mark.asyncio
    async def test_batch_streaming_single_batch(self) -> None:
        """Test handling of texts that fit in a single batch."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "dense_vecs": [[0.1] * 1024] * 10,
            "sparse_vecs": [{1: 0.5}] * 10,
            "latency_ms": 20.0,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        texts = [f"Text {i}" for i in range(10)]

        dense, sparse, latency = await embed_texts_in_batches(
            client=mock_client,
            texts=texts,
            embed_api_url="http://localhost:8001",
            batch_size=64,
        )

        # Should make only 1 request
        assert mock_client.post.call_count == 1
        assert len(dense) == 10
        assert len(sparse) == 10

    @pytest.mark.asyncio
    async def test_batch_streaming_empty_texts(self) -> None:
        """Test handling of empty text list."""
        mock_client = AsyncMock()

        dense, sparse, latency = await embed_texts_in_batches(
            client=mock_client,
            texts=[],
            embed_api_url="http://localhost:8001",
            batch_size=64,
        )

        # Should make no requests
        assert mock_client.post.call_count == 0
        assert dense == []
        assert sparse == []
        assert latency == 0.0

    def test_default_batch_size(self) -> None:
        """Test that default batch size is 64."""
        assert DEFAULT_EMBED_BATCH_SIZE == 64

    @pytest.mark.asyncio
    async def test_batch_streaming_correct_api_url(self) -> None:
        """Test that correct API URL is used in requests."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "dense_vecs": [[0.1] * 1024],
            "sparse_vecs": [{1: 0.5}],
            "latency_ms": 10.0,
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        await embed_texts_in_batches(
            client=mock_client,
            texts=["Test text"],
            embed_api_url="http://custom-host:9999",
            batch_size=64,
        )

        # Verify correct URL was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://custom-host:9999/embed"
