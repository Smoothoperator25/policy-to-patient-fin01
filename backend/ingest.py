"""PDF ingestor: per-page text extraction with pdfplumber, word bboxes, and
clause-aware section-heading detection.

Responsibilities
----------------
* Extract raw text per page and word-level bounding boxes ([x0, top, x1, bottom]).
* Detect section headings with the regex  ^\d+(\.\d+)*\s+\S
* Chunk by clause/paragraph: target 60–220 words, never merge across pages.
* Yield a ``ChunkBase`` pydantic instance per chunk (see ``backend/models.py``).
* If a PDF yields no readable text, return a clear error message.

Public API
----------
``extract_chunks(pdf_path: str, doc_id: str) -> Tuple[List[ChunkBase], str]``

Returns
-------
* A list of ``ChunkBase`` instances (may be empty if the PDF has no text).
* An error string (empty when no error). If the PDF has no readable text the
  error string is set to ``"No readable text. Scanned policies need OCR (planned)."``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pdfplumber

from backend.chunking import chunk_by_clauses
from backend.models import ChunkBase


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# Match lines like: "4.2 Definitions" or "1.1 Covered Services" or "12 Sub-limits"
_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+\S", re.MULTILINE)


def _detect_headings(text: str) -> List[int]:
    """Return a list of line indices that look like section headings.

    The regex ``^\d+(\.\d+)*\s+\S`` catches things like:
        "1.1 Covered Services"
        "4.2 Exclusions"
        "12 Deductible"
    but skips body paragraphs that start with numbers (e.g. "1) first").
    """
    lines = text.splitlines()
    heading_indices: List[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip very long lines that are clearly body text
        if len(stripped) > 120:
            continue
        if _HEADING_RE.match(stripped):
            heading_indices.append(i)
    return heading_indices


# ---------------------------------------------------------------------------
# Word-level bbox extraction
# ---------------------------------------------------------------------------

def _extract_words_with_bboxes(page) -> List[dict]:
    """Extract word dicts from a pdfplumber ``page`` object.

    pdfplumber's ``page.extract_words()`` returns a list of dicts with keys
    ``"x0", "top", "x1", "bottom", "text"``.  We normalise the output and
    guarantee the expected fields.
    """
    words = page.extract_words(
        strategy="dict",
        extra_attributes=["fontsize", "flags"],
    )
    normalized: List[dict] = []
    for w in words:
        if not w.get("text"):
            continue
        normalized.append(
            {
                "text": w["text"].strip(),
                "x0": float(w.get("x0", 0)),
                "top": float(w.get("top", 0)),
                "x1": float(w.get("x1", 0)),
                "bottom": float(w.get("bottom", 0)),
                "fontsize": float(w.get("fontsize", 10)),
                "flags": int(w.get("flags", 0)),
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Main entry point – chunking is delegated to backend.chunking
# ---------------------------------------------------------------------------

def extract_chunks(
    pdf_path: str,
    doc_id: str,
) -> Tuple[List[ChunkBase], str]:
    """Extract per-page text, word bboxes, and clause-aware chunks from a PDF.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.
    doc_id : str
        Identifier for the document (typically the filename without extension).

    Returns
    -------
    Tuple[List[ChunkBase], str]
        * ``List[ChunkBase]`` — may be empty if no text found.
        * ``str`` — error message (empty if no error). If the PDF contains no
          readable text the string is ``"No readable text. Scanned policies need
          OCR (planned)."``.
    """
    error_msg: str = ""

    # Try opening the PDF with pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
    except Exception as e:
        return [], f"Could not open PDF: {e}"

    if total_pages == 0:
        return [], "PDF has no pages."

    chunks: List[ChunkBase] = []

    for page_idx, page in enumerate(pdf.pages, start=1):
        # Extract raw text
        text = page.text or ""
        page_width = float(page.width or 0)
        page_height = float(page.height or 0)

        # If the entire PDF has no extractable text, surface the OCR notice
        if total_pages == 1 and not text.strip():
            error_msg = "No readable text. Scanned policies need OCR (planned)."
            # Still return an empty chunk list with the error
            return [], error_msg

        # Detect section headings on this page
        heading_indices = _detect_headings(text)
        section: str | None = None
        if heading_indices:
            # Use the first heading found as the section label
            # (in a full implementation we'd map each chunk to its nearest heading)
            section = text.splitlines()[heading_indices[0]].strip()

        # Extract word bboxes
        words = _extract_words_with_bboxes(page)
        # Build bboxes list in the required [[x0, top, x1, bottom], ...] shape
        bboxes: List[List[float]] = [
            [w["x0"], w["top"], w["x1"], w["bottom"]] for w in words
        ]

        # Chunk by clauses/paragraphs via the shared chunking module
        clause_chunks = chunk_by_clauses(
            text, page_idx, doc_id, section,
            word_min=60, word_max=220,
        )

        for chunk_dict in clause_chunks:
            # Attach page-level bboxes and dimensions to the chunk dict
            chunk_dict["bboxes"] = bboxes
            chunk_dict["page_width"] = page_width
            chunk_dict["page_height"] = page_height

            # Build the ChunkBase pydantic model instance
            chunk = ChunkBase(**chunk_dict)
            chunks.append(chunk)

    if not chunks and not error_msg:
        error_msg = "No readable text. Scanned policies need OCR (planned)."

    return chunks, error_msg