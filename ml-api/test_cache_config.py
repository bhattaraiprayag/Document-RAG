"""Unit tests for cache path configuration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cache_config import DEFAULT_LOCAL_CACHE_DIR, resolve_models_cache_dir


def test_resolve_models_cache_dir_uses_hf_home(tmp_path, monkeypatch):
    """Use HF_HOME when explicitly provided."""
    configured_dir = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(configured_dir))

    resolved = resolve_models_cache_dir()

    assert resolved == configured_dir
    assert resolved.is_dir()


def test_resolve_models_cache_dir_falls_back_to_project_local(monkeypatch):
    """Fallback to project-local cache path when HF_HOME is unset."""
    monkeypatch.delenv("HF_HOME", raising=False)

    resolved = resolve_models_cache_dir()

    assert resolved == DEFAULT_LOCAL_CACHE_DIR
    assert resolved.is_dir()


def test_resolve_models_cache_dir_raises_on_unwritable_dir(monkeypatch):
    """Raise a clear error when cache directory is not writable."""
    monkeypatch.setenv("HF_HOME", "/tmp/unwritable-cache")

    with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
        with pytest.raises(RuntimeError, match="not writable"):
            resolve_models_cache_dir()
