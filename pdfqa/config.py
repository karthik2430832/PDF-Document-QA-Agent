"""Runtime configuration, loaded from the environment with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    """Settings for the Q&A engine."""

    api_key: str

    # Gemini Models
    chat_model: str = "gemini-2.5-flash"

    # Chunking
    chunk_size: int = 1200
    overlap: int = 200

    # Retrieval
    top_k: int = 5

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
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set.\n"
                "Copy .env.example to .env and add your key, or export GEMINI_API_KEY.\n"
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        env_overrides = {}
        if os.environ.get("PDFQA_CHAT_MODEL"):
            env_overrides["chat_model"] = os.environ["PDFQA_CHAT_MODEL"]
        env_overrides.update(overrides)
        return cls(api_key=api_key, **env_overrides)
