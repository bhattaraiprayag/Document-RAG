"""Model factory for switching between LLM providers."""

import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, AsyncIterator

import httpx
from openai import AsyncOpenAI

from ..config import settings


class ModelFactoryError(Exception):
    """Exception raised for model factory errors."""

    pass


class ModelProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Yields:
            Token strings from the LLM response
        """
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI API provider implementation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model identifier (e.g., "gpt-4-turbo")
            base_url: API base URL
        """
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:  # type: ignore
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            error_msg = f"OpenAI API error: {type(e).__name__}"
            if hasattr(e, "message"):
                error_msg += f" - {e.message}"
            elif hasattr(e, "body"):
                error_msg += f" - {e.body}"
            else:
                error_msg += f" - {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(error_msg) from e


class OllamaProvider(ModelProvider):
    """Ollama API provider implementation."""

    def __init__(self, base_url: str, model: str) -> None:
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama server URL (e.g., "http://192.168.1.19:11434")
            model: Model identifier (e.g., "qwen3:30b-a3b")
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Increase timeout and configure for better connection handling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            follow_redirects=True,
        )

    async def generate_streaming(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama."""
        try:
            print(f"DEBUG: Connecting to Ollama at {self.base_url}/api/chat")
            print(f"DEBUG: Using model: {self.model}")

            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as response:
                print(f"DEBUG: Got response status: {response.status_code}")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
        except Exception as e:
            print(f"ERROR in Ollama streaming: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            raise


class ModelFactory:
    """Factory for creating LLM provider instances."""

    def __init__(self) -> None:
        """Initialize model factory with settings."""
        self.provider_type = settings.default_provider
        self.model_name = settings.default_model

    def get_provider(self) -> ModelProvider:
        """
        Get the configured LLM provider instance.

        Returns:
            ModelProvider instance (OpenAI or Ollama)

        Raises:
            ModelFactoryError: If provider type is invalid
        """
        if self.provider_type == "openai":
            if not settings.openai_api_key:
                raise ModelFactoryError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
                )
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                model=self.model_name,
                base_url=settings.openai_base_url,
            )

        elif self.provider_type == "ollama":
            return OllamaProvider(
                base_url=settings.ollama_base_url,
                model=self.model_name,
            )

        else:
            raise ModelFactoryError(
                f"Invalid provider: {self.provider_type}. Must be 'openai' or 'ollama'"
            )
