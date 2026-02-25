"""Application configuration."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_root_env_file() -> str:
    """Resolve the repository-root .env path for source-tree execution."""
    config_file = Path(__file__).resolve()
    for parent in config_file.parents:
        if (parent / "docker-compose.yml").exists() and (parent / "backend").exists():
            return str(parent / ".env")
    return str(Path.cwd() / ".env")


REPO_ROOT_ENV_FILE = Path(_resolve_root_env_file())
ENV_FILE = str(REPO_ROOT_ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
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
