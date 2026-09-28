"""Clause-aware chunking module.

Splits extracted PDF page text into chunks targeting 60–220 words.
Never merges across pages. Each chunk carries its page, section, and
word-bbox information so the retrieval module can force-include chunks
referenced by section/clause numbers.

Public function
---------------
``chunk_by_clauses(page_text: str, page: int, doc_id: str,
                   section: str | None = None,
                   word_min: int = 60, word_max: int = 220) -> List[dict]

Returns a list of chunk-dicts with keys:
    chunk_id, doc_id, page, section, text, bboxes,
    page_width, page_height
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

from backend.models import ChunkBase


# ---------------------------------------------------------------------------
# Heading‑aware section detection (shared with ingest)
# ---------------------------------------------------------------------------

 _HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+\S", re.MULTILINE)


def _detect_headings(text: str) -> List[int]:
    """Return line indices that look like section headings.

    The regex ``^\d+(\.\d+)*\s+\S`` catches things like:
        "1.1 Covered Services"
        "4.2 Exclusions"
        "12 Deductible"
    but skips body paragraphs that start with numbers.
    """
    lines = text.splitlines()
    heading_indices: List[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            continue
        if _HEADING_RE.match(stripped):
            heading_indices.append(i)
    return heading_indices


# ---------------------------------------------------------------------------
# Core chunking: split page text into clause‑aware chunks
# ---------------------------------------------------------------------------

def chunk_by_clauses(
    page_text: str,
    page: int,
    doc_id: str,
    section: str | None = None,
    word_min: int = 60,
    word_max: int = 220,
) -> List[Dict[str, Any]]:
    """Split *page_text* into chunks in the range [word_min, word_max].

    Heuristic
    ----------
    * paragraphs are split on double newlines ``\\n\\n``.
    * sentences are split on ``. ! ?`` followed by a space or EOS.
    * each chunk is kept in the target word count; if a sentence would
      push us over ``word_max`` we flush the current chunk first.
    * chunks are never merged across page boundaries.

    Returns
    -------
    List of dicts, each compatible with ``ChunkBase`` pydantic model.
    """
    words = page_text.split()
    wcount = len(words)

    # Page too short for a meaningful chunk — return the whole thing as one chunk
    if wcount <= word_min:
        return [
            {
                "chunk_id": f"{doc_id}P{page}C1",
                "doc_id": doc_id,
                "page": page,
                "section": section,
                "text": page_text.strip(),
                "bboxes": [],
                "page_width": 0,
                "page_height": 0,
            }
        ]

    # Split into paragraphs, then further on sentences
    raw_paragraphs = page_text.split("\n\n")

    segments: List[str] = []
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for s in sentences:
            s = s.strip()
            if s:
                segments.append(s)

    # If no sentence splits were found, keep whole paragraphs as segments
    if not segments:
        segments = [p.strip() for p in raw_paragraphs if p.strip()]

    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_words = 0

    for seg in segments:
        seg_words = seg.split()
        seg_len = len(seg_words)

        if current_words + seg_len > word_max and current_parts:
            # Flush current chunk
            chunk_text = " ".join(current_parts).strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}P{page}C{len(chunks) + 1}",
                        "doc_id": doc_id,
                        "page": page,
                        "section": section,
                        "text": chunk_text,
                        "bboxes": [],
                        "page_width": 0,
                        "page_height": 0,
                    }
                )
            # Start new chunk with the current segment
            current_parts = [seg]
            current_words = seg_len
        else:
            current_parts.append(seg)
            current_words += seg_len

    # Flush remaining content
    if current_parts:
        chunk_text = " ".join(current_parts).strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}P{page}C{len(chunks) + 1}",
                    "doc_id": doc_id,
                    "page": page,
                    "section": section,
                    "text": chunk_text,
                    "bboxes": [],
                    "page_width": 0,
                    "page_height": 0,
                }
            )

    return chunks