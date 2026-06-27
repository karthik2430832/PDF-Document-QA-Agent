"""Page-aware PDF text extraction using PyMuPDF.

PyMuPDF preserves layout/reading order far better than PyPDF2, which matters
both for answer quality and for keeping accurate page numbers for citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf


@dataclass
class PageText:
    page: int  # 1-based page number
    text: str


def extract_pages(pdf_path: str | None = None, *, data: bytes | None = None) -> list[PageText]:
    """Extract text per page from a PDF on disk or from raw bytes.

    Pass either ``pdf_path`` (a file path) or ``data`` (the PDF bytes, e.g. from
    a Streamlit upload). Returns one ``PageText`` per page, in order.
    """
    if data is not None:
        doc = pymupdf.open(stream=data, filetype="pdf")
    elif pdf_path is not None:
        doc = pymupdf.open(pdf_path)
    else:
        raise ValueError("extract_pages requires either pdf_path or data")

    try:
        return [
            PageText(page=i + 1, text=doc[i].get_text("text") or "")
            for i in range(doc.page_count)
        ]
    finally:
        doc.close()
