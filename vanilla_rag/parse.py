"""Document parsing.

Using pumypdf.
"""

from pathlib import Path

import pymupdf

TEXT_SUFFIXES = {".txt", ".md", ".rst"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


def extract_text(path: str | Path) -> str:
    """Return the raw text of a document."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def extract_pdf_text(path: str | Path) -> str:
    """Extract raw text from a PDF with PyMuPDF, page by page."""
    with pymupdf.open(str(path)) as doc:
        pages = [page.get_text("text") for page in doc]
    return "\n\n".join(pages)
