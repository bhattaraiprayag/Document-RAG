"""Unit tests for model factory."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models.model_factory import (
    ModelFactory,
    ModelFactoryError,
    OllamaProvider,
    OpenAIProvider,
)


@pytest.mark.unit
class TestModelFactory:
    """Test model factory creation and provider selection."""

    def test_factory_creates_openai_provider(self) -> None:
        """Test factory creates OpenAI provider when configured."""
        with patch("app.models.model_factory.settings") as mock_settings:
            mock_settings.default_provider = "openai"
            mock_settings.default_model = "gpt-4"
            mock_settings.openai_api_key = "test-key"
            mock_settings.openai_base_url = "https://api.openai.com/v1"

            factory = ModelFactory()
            provider = factory.get_provider()
            assert isinstance(provider, OpenAIProvider)

    def test_factory_creates_ollama_provider(self) -> None:
        """Test factory creates Ollama provider when configured."""
        with patch("app.models.model_factory.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.default_model = "qwen3:30b-a3b"
            mock_settings.ollama_base_url = "http://localhost:11434"

            factory = ModelFactory()
            provider = factory.get_provider()
            assert isinstance(provider, OllamaProvider)

    def test_factory_raises_on_invalid_provider(self) -> None:
        """Test factory raises error for invalid provider."""
        with patch("app.models.model_factory.settings") as mock_settings:
            mock_settings.default_provider = "invalid"
            mock_settings.default_model = "test-model"

            factory = ModelFactory()
            with pytest.raises(ModelFactoryError):
                factory.get_provider()


@pytest.mark.unit
class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    @pytest.mark.asyncio
    async def test_generate_streaming(self) -> None:
        """Test OpenAI streaming generation."""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" world"))]),
        ]
        mock_client.chat.completions.create.return_value = mock_stream

        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        provider.client = mock_client

        tokens = []
        async for token in provider.generate_streaming(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate_streaming_handles_empty_content(self) -> None:
        """Test OpenAI provider handles None content in stream."""
        mock_client = AsyncMock()
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = [
            Mock(choices=[Mock(delta=Mock(content=None))]),
            Mock(choices=[Mock(delta=Mock(content="Hi"))]),
        ]
        mock_client.chat.completions.create.return_value = mock_stream

        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        provider.client = mock_client

        tokens = []
        async for token in provider.generate_streaming(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["Hi"]


@pytest.mark.unit
class TestOllamaProvider:
    """Test Ollama provider implementation."""

    @pytest.mark.asyncio
    async def test_generate_streaming(self) -> None:
        """Test Ollama streaming generation."""

        async def mock_aiter_lines():
            lines = [
                '{"message": {"content": "Hello"}}',
                '{"message": {"content": " world"}}',
                '{"done": true}',
            ]
            for line in lines:
                yield line

        mock_response = Mock()
        mock_response.aiter_lines = Mock(return_value=mock_aiter_lines())
        mock_response.raise_for_status = Mock()

        # Create async context manager mock
        class MockStream:
            async def __aenter__(self):
                return mock_response

            async def __aexit__(self, *args):
                return None

        mock_client = Mock()
        mock_client.stream = Mock(return_value=MockStream())

        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3")
        provider.client = mock_client

        tokens = []
        async for token in provider.generate_streaming(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate_streaming_handles_malformed_json(self) -> None:
        """Test Ollama provider handles malformed JSON gracefully."""

        async def mock_aiter_lines():
            lines = [
                '{"message": {"content": "Hello"}}',
                "invalid json",
                '{"done": true}',
            ]
            for line in lines:
                yield line

        mock_response = Mock()
        mock_response.aiter_lines = Mock(return_value=mock_aiter_lines())
        mock_response.raise_for_status = Mock()

        # Create async context manager mock
        class MockStream:
            async def __aenter__(self):
                return mock_response

            async def __aexit__(self, *args):
                return None

        mock_client = Mock()
        mock_client.stream = Mock(return_value=MockStream())

        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3")
        provider.client = mock_client

        tokens = []
        async for token in provider.generate_streaming(
            messages=[{"role": "user", "content": "Hi"}]
        ):
            tokens.append(token)

        # Should skip malformed line
        assert tokens == ["Hello"]
