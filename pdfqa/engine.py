"""The retrieval-augmented Q&A engine powered by Google Gemini.

Pipeline: extract -> chunk -> index (TF-IDF + Cosine, cached) -> retrieve top-k -> generate with Gemini.
Answers stream token-by-token and are grounded in the retrieved excerpts,
which are labelled with page numbers so the model can cite them.

Zero native DLL / C++ redistributable dependencies — 100% portable on any OS/architecture.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import re
import time
from collections import Counter
from typing import Iterator, Optional, Sequence

import numpy as np
from google import genai
from google.genai import errors, types

from .chunking import Chunk, chunk_pages
from .config import Config
from .extract import extract_pages

logger = logging.getLogger("pdfqa")

SYSTEM_INSTRUCTION = (
    "You are a precise, helpful assistant that answers questions about an uploaded PDF document. "
    "Use ONLY the provided context excerpts to answer. Each excerpt is labelled "
    "with the page number(s) it came from. When you use information from an "
    "excerpt, cite the page(s) inline like (p. 3) or (pp. 4-5). If the answer cannot be found "
    "in the excerpts, politely and clearly state that the information isn't present in the document "
    "— do not fabricate facts or invent details."
)


class PdfQAEngine:
    """Index a PDF and answer questions about it with exact citations using Gemini."""

    def __init__(self, config: Config):
        self.cfg = config
        self.client = genai.Client(api_key=config.api_key)
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # Normalized TF-IDF matrix
        self.vocab: dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.doc_name: Optional[str] = None

    # ------------------------------------------------------------------ build

    def build(
        self,
        pdf_path: str | None = None,
        *,
        data: bytes | None = None,
        name: str | None = None,
    ) -> "PdfQAEngine":
        """Extract, chunk and index a PDF (using cache when possible)."""
        if data is None:
            if pdf_path is None:
                raise ValueError("build requires either pdf_path or data")
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            with open(pdf_path, "rb") as fh:
                data = fh.read()

        self.doc_name = name or (os.path.basename(pdf_path) if pdf_path else "document")

        signature = self._signature(data)
        if self.cfg.use_cache and self._load_cache(signature):
            logger.info("Loaded %d chunks from cache.", len(self.chunks))
            return self

        pages = extract_pages(data=data)
        self.chunks = chunk_pages(pages, self.cfg.chunk_size, self.cfg.overlap)
        if not self.chunks:
            raise ValueError(
                "No extractable text found in the PDF. It may be a scanned/image "
                "document that needs OCR."
            )

        logger.info("Indexing %d chunks ...", len(self.chunks))
        self._build_index([c.text for c in self.chunks])
        if self.cfg.use_cache:
            self._save_cache(signature)
        return self

    # -------------------------------------------------------------- retrieval

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return the most relevant chunks for a query using cosine similarity."""
        if not self.chunks or self.embeddings is None:
            return []
        k = top_k or self.cfg.top_k
        if len(self.chunks) <= k:
            return list(self.chunks)

        query_vec = self._vectorize(query)
        # Cosine similarity: both embeddings and query_vec are L2-normalized
        scores = self.embeddings @ query_vec

        # If all scores are zero (e.g. query terms not in vocabulary), return top k by document order
        if np.all(scores == 0):
            return list(self.chunks[:k])

        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]

    # ------------------------------------------------------------- generation

    def answer_stream(
        self,
        query: str,
        chunks: Sequence[Chunk],
        history: Optional[list[dict]] = None,
    ) -> Iterator[str]:
        """Yield answer text token-by-token from Gemini, grounded in ``chunks``."""
        prompt = (
            "Context excerpts from the document:\n\n"
            f"{self._format_context(chunks)}\n\n"
            f"Question: {query}"
        )
        contents = self._build_contents(history, prompt)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.cfg.temperature,
        )

        for attempt in range(self.cfg.retries):
            started = False
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.cfg.chat_model,
                    contents=contents,
                    config=config,
                )
                for event in stream:
                    if event.text:
                        started = True
                        yield event.text
                return
            except errors.ServerError:
                if started or attempt == self.cfg.retries - 1:
                    raise
                wait = 2 ** attempt
                logger.info("Gemini busy, retrying in %ss ...", wait)
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001
                if attempt == self.cfg.retries - 1:
                    raise
                wait = 2 ** attempt
                logger.info("Gemini API error, retrying in %ss: %s", wait, exc)
                time.sleep(wait)

    def answer(
        self,
        query: str,
        chunks: Sequence[Chunk],
        history: Optional[list[dict]] = None,
    ) -> str:
        """Convenience wrapper that collects the streamed answer into a string."""
        return "".join(self.answer_stream(query, chunks, history))

    # ------------------------------------------------------------- internal TF-IDF

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric token extraction."""
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def _build_index(self, texts: list[str]) -> None:
        """Compute vocabulary, IDF statistics, and L2-normalized document matrix."""
        tokenized_docs = [self._tokenize(t) for t in texts]
        n_docs = len(tokenized_docs)

        # Build vocabulary
        vocab: dict[str, int] = {}
        for doc in tokenized_docs:
            for token in set(doc):
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab = vocab
        vocab_size = len(vocab)

        if vocab_size == 0:
            self.embeddings = np.zeros((n_docs, 1), dtype=np.float32)
            self.idf = np.ones((1,), dtype=np.float32)
            return

        # Document frequencies
        df = np.zeros(vocab_size, dtype=np.float32)
        tf_matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)

        for i, doc in enumerate(tokenized_docs):
            counts = Counter(doc)
            total = len(doc) or 1
            for token, count in counts.items():
                idx = vocab[token]
                tf_matrix[i, idx] = count / total
                df[idx] += 1

        # Smooth IDF: ln((1 + n) / (1 + df)) + 1
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        # Compute TF-IDF matrix
        tfidf = tf_matrix * self.idf

        # L2 normalize rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (tfidf / norms).astype(np.float32)

    def _vectorize(self, text: str) -> np.ndarray:
        """Convert a query string to a normalized TF-IDF vector."""
        if not self.vocab or self.idf is None:
            return np.zeros((1,), dtype=np.float32)

        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        if not tokens:
            return vec

        counts = Counter(tokens)
        total = len(tokens)
        for token, count in counts.items():
            if token in self.vocab:
                vec[self.vocab[token]] = (count / total) * self.idf[self.vocab[token]]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _format_context(chunks: Sequence[Chunk]) -> str:
        blocks = []
        for c in chunks:
            label = (
                f"Page {c.page_start}"
                if c.page_start == c.page_end
                else f"Pages {c.page_start}-{c.page_end}"
            )
            blocks.append(f"[{label}]\n{c.text}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _build_contents(history: Optional[list[dict]], prompt: str) -> list[types.Content]:
        contents: list[types.Content] = []
        for turn in history or []:
            contents.append(
                types.Content(role="user", parts=[types.Part(text=turn["q"])])
            )
            contents.append(
                types.Content(role="model", parts=[types.Part(text=turn["a"])])
            )
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return contents

    # ----------------------------------------------------------------- cache

    def _signature(self, data: bytes) -> str:
        h = hashlib.sha256()
        h.update(data)
        h.update(
            f"gemini_tfidf|{self.cfg.chunk_size}|{self.cfg.overlap}".encode()
        )
        return h.hexdigest()[:16]

    def _cache_file(self, signature: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"{signature}.pkl")

    def _load_cache(self, signature: str) -> bool:
        path = self._cache_file(signature)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            self.chunks = payload["chunks"]
            self.embeddings = payload["embeddings"]
            self.vocab = payload.get("vocab", {})
            self.idf = payload.get("idf")
            return True
        except Exception:  # corrupt/old cache — rebuild
            logger.warning("Ignoring unreadable cache at %s", path)
            return False

    def _save_cache(self, signature: str) -> None:
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        with open(self._cache_file(signature), "wb") as fh:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "embeddings": self.embeddings,
                    "vocab": self.vocab,
                    "idf": self.idf,
                },
                fh,
            )
