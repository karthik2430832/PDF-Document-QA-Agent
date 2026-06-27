"""The retrieval-augmented Q&A engine.

Pipeline: extract -> chunk -> embed (cached) -> retrieve top-k -> generate.
Answers stream token-by-token and are grounded in the retrieved excerpts,
which are labelled with page numbers so the model can cite them.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from typing import Iterator, Optional, Sequence

import numpy as np
from google import genai
from google.genai import errors, types

from .chunking import Chunk, chunk_pages
from .config import Config
from .extract import extract_pages

logger = logging.getLogger("pdfqa")

SYSTEM_INSTRUCTION = (
    "You are a precise assistant that answers questions about a PDF document. "
    "Use ONLY the provided context excerpts to answer. Each excerpt is labelled "
    "with the page number(s) it came from. When you use information from an "
    "excerpt, cite the page(s) inline like (p. 3). If the answer cannot be found "
    "in the excerpts, say you couldn't find it in the document — do not invent "
    "details."
)


class PdfQAEngine:
    """Index a PDF and answer questions about it with citations."""

    def __init__(self, config: Config):
        self.cfg = config
        self.client = genai.Client(api_key=config.api_key)
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.doc_name: Optional[str] = None

    # ------------------------------------------------------------------ build

    def build(
        self,
        pdf_path: str | None = None,
        *,
        data: bytes | None = None,
        name: str | None = None,
    ) -> "PdfQAEngine":
        """Extract, chunk and embed a PDF (using the cache when possible)."""
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

        logger.info("Embedding %d chunks ...", len(self.chunks))
        self.embeddings = self._embed(
            [c.text for c in self.chunks], task_type="RETRIEVAL_DOCUMENT"
        )
        if self.cfg.use_cache:
            self._save_cache(signature)
        return self

    # -------------------------------------------------------------- retrieval

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return the most relevant chunks for a query (cosine similarity)."""
        if not self.chunks:
            return []
        k = top_k or self.cfg.top_k
        if len(self.chunks) <= k:
            return list(self.chunks)

        query_vec = self._embed([query], task_type="RETRIEVAL_QUERY")[0]
        scores = self.embeddings @ query_vec  # both are L2-normalised -> cosine
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]

    # ------------------------------------------------------------- generation

    def answer_stream(
        self,
        query: str,
        chunks: Sequence[Chunk],
        history: Optional[list[dict]] = None,
    ) -> Iterator[str]:
        """Yield answer text token-by-token, grounded in ``chunks``.

        ``history`` is a list of ``{"q": ..., "a": ...}`` prior turns.
        """
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
                    model=self.cfg.chat_model, contents=contents, config=config
                )
                for event in stream:
                    if event.text:
                        started = True
                        yield event.text
                return
            except errors.ServerError:
                # If tokens already streamed we can't cleanly restart — surface it.
                if started or attempt == self.cfg.retries - 1:
                    raise
                wait = 2 ** attempt
                logger.info("Model busy, retrying in %ss ...", wait)
                time.sleep(wait)

    def answer(
        self,
        query: str,
        chunks: Sequence[Chunk],
        history: Optional[list[dict]] = None,
    ) -> str:
        """Convenience wrapper that collects the streamed answer into a string."""
        return "".join(self.answer_stream(query, chunks, history))

    # --------------------------------------------------------------- internal

    def _embed(self, texts: list[str], *, task_type: str) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.embed_batch):
            batch = texts[i : i + self.cfg.embed_batch]
            response = self._with_retry(
                lambda b=batch: self.client.models.embed_content(
                    model=self.cfg.embed_model,
                    contents=b,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.cfg.embed_dim,
                    ),
                ),
                what="embedding",
            )
            vectors.extend(e.values for e in response.embeddings)

        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _with_retry(self, call, *, what: str):
        for attempt in range(self.cfg.retries):
            try:
                return call()
            except errors.ServerError:
                if attempt == self.cfg.retries - 1:
                    raise
                wait = 2 ** attempt
                logger.info("%s busy, retrying in %ss ...", what.capitalize(), wait)
                time.sleep(wait)

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
            f"{self.cfg.embed_model}|{self.cfg.embed_dim}|"
            f"{self.cfg.chunk_size}|{self.cfg.overlap}".encode()
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
            return True
        except Exception:  # corrupt/old cache — rebuild
            logger.warning("Ignoring unreadable cache at %s", path)
            return False

    def _save_cache(self, signature: str) -> None:
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        with open(self._cache_file(signature), "wb") as fh:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, fh)
