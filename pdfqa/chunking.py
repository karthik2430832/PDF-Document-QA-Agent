"""Split extracted pages into overlapping chunks that remember their pages.

Chunks are sized in characters with a sliding-window overlap so context that
straddles a chunk boundary isn't lost. Each chunk records the page range it
came from, which is what powers the page citations in answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .extract import PageText


@dataclass
class Chunk:
    id: int
    text: str
    page_start: int
    page_end: int


def chunk_pages(pages: list[PageText], chunk_size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Concatenate pages and slice into overlapping chunks with page ranges."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Build one text blob while remembering where each page begins.
    parts: list[str] = []
    page_starts: list[tuple[int, int]] = []  # (char_offset, page_number)
    cursor = 0
    for p in pages:
        text = (p.text or "").strip()
        if not text:
            continue
        page_starts.append((cursor, p.page))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" join separator

    full_text = "\n".join(parts)
    if not full_text:
        return []

    def page_at(char_idx: int) -> int:
        page = page_starts[0][1]
        for start, number in page_starts:
            if start <= char_idx:
                page = number
            else:
                break
        return page

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    cid = 0
    for start in range(0, len(full_text), step):
        piece = full_text[start : start + chunk_size]
        if piece.strip():
            end = start + len(piece) - 1
            chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    page_start=page_at(start),
                    page_end=page_at(end),
                )
            )
            cid += 1
        if start + chunk_size >= len(full_text):
            break
    return chunks


def pages_label(chunks: Iterable[Chunk]) -> str:
    """Render a deduplicated, sorted page list like ``p.1, p.3, p.4``."""
    pages: set[int] = set()
    for c in chunks:
        pages.update(range(c.page_start, c.page_end + 1))
    return ", ".join(f"p.{p}" for p in sorted(pages))
