"""Pytest configuration and shared fixtures."""
import os
import pytest
from typing import Generator
from dotenv import load_dotenv

# Load test environment
load_dotenv(".env.test")


@pytest.fixture(scope="session")
def test_env() -> dict[str, str]:
    """Test environment variables."""
    return {
        "DEFAULT_PROVIDER": "ollama",
        "DEFAULT_MODEL": "qwen3:30b-a3b",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "QDRANT_URL": "http://localhost:6333",
    }


@pytest.fixture
def sample_markdown() -> str:
    """Sample markdown content for testing."""
    return """# Chapter 1: Introduction

## 1.1 Background
This is the background section with some content that spans multiple sentences.
We need enough text here to test chunking properly.

## 1.2 Methodology
This section describes the methodology used in the research.
It also has multiple sentences for testing purposes.

# Chapter 2: Results
The results section contains findings from the research.
"""


@pytest.fixture
def sample_file_hash() -> str:
    """Sample file hash for testing."""
    return "abc123def456789"
