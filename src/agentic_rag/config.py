"""Centralized runtime configuration pulled from environment variables.

All environment variables are read once at import time via :py:func:`Settings`.
Tests and callers should reference :data:`settings` rather than calling
``os.getenv`` directly so configuration stays discoverable and overridable in
one place. A ``.env`` file in the working directory is auto-loaded by
python-dotenv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    """Return the string env var ``key`` or ``default`` if unset."""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """Return the int env var ``key`` or ``default`` if unset / unparseable."""
    return int(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    """Return the bool env var ``key`` (truthy strings: 1/true/yes/on)."""
    return os.getenv(key, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable, env-driven runtime configuration.

    The default values are tuned for the local prototype: Ollama on
    ``localhost:11434`` with ``qwen2.5:7b-instruct``, BGE-small embeddings,
    indices under ``./data/``. Override any field by exporting the matching
    environment variable (see ``.env.example``).
    """

    # LLM
    llm_backend: str = _env("LLM_BACKEND", "ollama")
    """``"ollama"`` for the real local LLM, ``"dummy"`` for the deterministic stub."""

    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    """Base URL of the Ollama HTTP server."""

    ollama_model: str = _env("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    """Ollama model tag pulled and served by the Ollama runner."""

    # Embeddings
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    """Sentence-transformer model identifier used to embed chunks and queries."""

    # Paths
    data_dir: Path = Path(_env("DATA_DIR", "./data"))
    """Root directory holding raw / processed data and built indices."""

    chroma_dir: Path = Path(_env("CHROMA_DIR", "./data/chroma"))
    """Chroma persistent client directory (file-backed)."""

    graph_path: Path = Path(_env("GRAPH_PATH", "./data/graph/citation_graph.pkl"))
    """Pickled NetworkX citation graph location."""

    processed_path: Path = Path(_env("PROCESSED_PATH", "./data/processed/chunks.jsonl"))
    """Canonical JSONL file emitted by the ingestion pipeline."""

    # Retrieval
    top_k_vector: int = _env_int("TOP_K_VECTOR", 8)
    """Top-K neighbours fetched per rewritten query in ``vector_retrieve``."""

    graph_hops: int = _env_int("GRAPH_HOPS", 1)
    """BFS depth from each vector seed during graph expansion."""

    top_k_final: int = _env_int("TOP_K_FINAL", 6)
    """Number of fused chunks kept after RRF and forwarded to the composer."""

    # Tools
    web_search_enabled: bool = _env_bool("WEB_SEARCH_ENABLED", True)
    """Master switch for the DuckDuckGo web-search tool."""


settings = Settings()
"""Process-wide singleton; import this rather than constructing a new ``Settings``."""
