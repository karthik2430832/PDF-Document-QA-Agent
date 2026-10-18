"""Page-aware PDF text extraction using pypdfium2.

pypdfium2 (Google's PDFium) preserves layout/reading order and ships native
wheels for Windows ARM64, avoiding the VC++ redistributable dependency that
pymupdf requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import pypdfium2


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
        doc = pypdfium2.PdfDocument(data)
    elif pdf_path is not None:
        doc = pypdfium2.PdfDocument(pdf_path)
    else:
        raise ValueError("extract_pages requires either pdf_path or data")

    try:
        pages = []
        for i in range(len(doc)):
            textpage = doc[i].get_textpage()
            text = textpage.get_text_range() or ""
            pages.append(PageText(page=i + 1, text=text))
        return pages
    finally:
        doc.close()
