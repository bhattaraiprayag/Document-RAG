"""Application configuration."""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields not defined in the model
    )

    # Model Provider Configuration
    default_provider: Literal["openai", "ollama"] = "ollama"
    default_model: str = "qwen3:30b-a3b"

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Ollama Configuration
    ollama_base_url: str = "http://192.168.1.19:11434"  # Fixed default to LAN IP

    # Service URLs
    qdrant_url: str = "http://localhost:6333"
    embed_api_url: str = "http://localhost:8001"
    rerank_api_url: str = "http://localhost:8001"  # Same as embed, unified ML API

    # HuggingFace
    hf_home: str = "./models_cache"

    # HTTP Client Configuration
    rag_http_timeout: float = 120.0  # Timeout for embed/rerank API calls in seconds


settings = Settings()
