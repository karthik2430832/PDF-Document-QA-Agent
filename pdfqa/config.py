"""Runtime configuration, loaded from the environment with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    """Settings for the Q&A engine.

    Anything here can be overridden via ``Config.from_env(top_k=8, ...)``.
    """

    api_key: str

    # Models (see https://ai.google.dev/gemini-api/docs/models)
    chat_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"

    # Chunking (character based — simple and dependency free)
    chunk_size: int = 1200
    overlap: int = 200

    # Retrieval / embeddings
    top_k: int = 5
    embed_dim: int = 768  # Matryoshka truncation; keeps the index small
    embed_batch: int = 50

    # Generation
    temperature: float = 0.2

    # Robustness
    retries: int = 4

    # On-disk embedding cache so re-runs over the same PDF are instant
    use_cache: bool = True
    cache_dir: str = ".pdfqa_cache"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config from a ``.env`` / environment, applying ``overrides``."""
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set.\n"
                "Copy .env.example to .env and add your key, or export GEMINI_API_KEY.\n"
                "Get a key at https://aistudio.google.com/apikey"
            )
        # Allow a couple of common env overrides without code changes.
        env_overrides = {}
        if os.environ.get("PDFQA_CHAT_MODEL"):
            env_overrides["chat_model"] = os.environ["PDFQA_CHAT_MODEL"]
        if os.environ.get("PDFQA_EMBED_MODEL"):
            env_overrides["embed_model"] = os.environ["PDFQA_EMBED_MODEL"]
        env_overrides.update(overrides)
        return cls(api_key=api_key, **env_overrides)
