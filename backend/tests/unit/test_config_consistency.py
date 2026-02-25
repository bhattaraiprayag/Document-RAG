"""Configuration consistency regression tests."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_env_example_uses_unified_ml_api_endpoint() -> None:
    """Embed and rerank should point to unified ml-api endpoint."""
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "EMBED_API_URL=http://localhost:8001" in env_example
    assert "RERANK_API_URL=http://localhost:8001" in env_example


def test_compose_backend_env_file_matches_quickstart_convention() -> None:
    """Compose backend env_file should align with quickstart setup steps."""
    compose_yml = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    quickstart = (REPO_ROOT / "QUICKSTART.md").read_text(encoding="utf-8")

    assert "env_file:\n      - ./.env" in compose_yml
    assert "cp .env.example .env" in quickstart
