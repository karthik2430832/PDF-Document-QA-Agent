"""PDF Document Q&A — a small RAG library over Google Gemini.

Importing this package installs the OS trust store into Python's SSL stack,
which is needed for HTTPS to work behind corporate TLS-inspecting proxies.
"""

try:  # best-effort; harmless if truststore is unavailable
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - environment dependent
    pass

from .chunking import Chunk, chunk_pages, pages_label
from .config import Config
from .engine import PdfQAEngine
from .extract import PageText, extract_pages

__all__ = [
    "Config",
    "PdfQAEngine",
    "PageText",
    "extract_pages",
    "Chunk",
    "chunk_pages",
    "pages_label",
]
