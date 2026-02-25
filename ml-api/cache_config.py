"""Cache path helpers for ML model artifacts."""

import os
from pathlib import Path

DEFAULT_LOCAL_CACHE_DIR = Path(__file__).resolve().parent / "models_cache"


def resolve_models_cache_dir() -> Path:
    """
    Resolve and validate a writable cache directory for model downloads.

    Priority:
    1. HF_HOME environment variable when explicitly provided
    2. Project-local ml-api/models_cache directory
    """
    configured_cache = os.getenv("HF_HOME")
    cache_dir = (
        Path(configured_cache).expanduser()
        if configured_cache
        else DEFAULT_LOCAL_CACHE_DIR
    )

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe_file = cache_dir / ".cache_write_probe"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Model cache directory '{cache_dir}' is not writable. "
            "Set HF_HOME to a writable path."
        ) from exc

    return cache_dir
