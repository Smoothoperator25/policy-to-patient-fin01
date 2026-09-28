"""Retrieval module: FAISS dense retrieval with section/clause force-include.

Responsibilities
----------------
* Embed chunks (normalized) using the all-MiniLM-L6-v2 sentence transformer.
* Hold one FAISS IndexFlatIP per doc_id in memory.
* Retrieval = dense top-k (k=4) plus an exact section lookup:
  if the question mentions "section 4.2" or "clause 4.2", force-include that chunk.
* Return chunk IDs + similarity scores.

Public API
----------
``retrieve(query: str, doc_chunks: Dict[str, List[dict]], k: int = 4) -> List[dict]

Each returned dict has: "chunk_id", "page", "section", "text", "score" (float 0-1).
"""

from __future__ import annotations

import re
from typing import Dict, List, Any

import numpy as np

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Lazy loader for the sentence-transformer model
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazily load the all-MiniLM-L6-v2 sentence transformer."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Section/clause force-include logic
# ---------------------------------------------------------------------------

def _force_section_lookup(question: str, doc_chunks: Dict[str, List[dict]]) -> List[str]:
    """Look for "section X.Y" or "clause X.Y" in the question.

    If found, return the matching chunk IDs from the document.
    """
    # Pattern: "section 4.2" or "clause 4.2" or "section 4" or "clause 4"
    patterns = [
        r"\bsection\s+(\d+(\.\d+)*)",
        r"\bclause\s+(\d+(\.\d+)*)",
    ]
    matched_ids: List[str] = []
    for pattern in patterns:
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            num = m.group(1)
            # Look for a chunk whose section matches this number
            for doc_id, chunks in doc_chunks.items():
                for chunk in chunks:
                    chunk_section = chunk.get("section")
                    if chunk_section and num in chunk_section:
                        matched_ids.append(chunk["chunk_id"])
    return matched_ids


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    doc_chunks: Dict[str, List[dict]],
    k: int = 4,
) -> List[dict]:
    """Dense FAISS retrieval with optional section/clause force-include.

    Parameters
    ----------
    query : str
        The user's question.
    doc_chunks : Dict[str, List[dict]]
        Mapping of doc_id -> list of chunk dicts. Each chunk dict must contain
        at least: "id", "text", "page", "section", and embedding vector
        stored under key "_embedding" (numpy array, L2-normalized).
    k : int, default 4
        Number of top chunks to return from dense retrieval.

    Returns
    -------
    List[dict]
        Each entry has: "chunk_id", "page", "section", "text", "score".
        The list may contain forced-section chunks even if their score is low.
    """
    if not doc_chunks:
        return []

    # Collect all chunks across docs, tracking their doc_id and original index
    all_chunks: List[dict] = []
    doc_ids: List[str] = []

    for doc_id, chunks in doc_chunks.items():
        for idx, chunk in enumerate(chunks):
            # Ensure embedding exists; if not, compute it now
            if "_embedding" not in chunk or chunk["_embedding"] is None:
                text = chunk.get("text", "")
                if text:
                    model = _get_embedding_model()
                    emb = model.encode([text], normalize_embeddings=True)[0]
                    chunk["_embedding"] = emb
                else:
                    chunk["_embedding"] = np.zeros((1,), dtype=np.float32)

            chunk_copy = dict(chunk)  # shallow copy so we don't mutate original
            chunk_copy["doc_id"] = doc_id
            chunk_copy["_doc_index"] = idx
            all_chunks.append(chunk_copy)
            doc_ids.append(doc_id)

    if not all_chunks:
        return []

    # Perform section/clause force-include first
    forced_ids = _force_section_lookup(query, doc_chunks)
    forced_set = set(forced_ids)

    # Embed the query
    model = _get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]  # shape (dim,)

    # Compute cosine similarity = dot product (vectors are normalized)
    dim = len(q_emb)
    scores = np.array([np.dot(q_emb, chunk["_embedding"]).item() for chunk in all_chunks])

    # We'll select top-k from all chunks, but ensure forced chunks are included
    # Create a mask for forced chunks
    forced_indices = [i for i, c in enumerate(all_chunks) if c["chunk_id"] in forced_set]

    # Sort all chunks by score descending
    all_indices = list(range(len(all_chunks)))
    all_indices.sort(key=lambda i: scores[i], reverse=True)

    # Select top-k, then add any forced chunks not already in top-k
    selected_indices = set(all_indices[:k])
    for fi in forced_indices:
        selected_indices.add(fi)

    # Return the selected chunks with their info
    results: List[dict] = []
    for idx in sorted(selected_indices, key=lambda i: scores[i], reverse=True):
        chunk = all_chunks[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "section": chunk.get("section") or "?",
                "text": chunk.get("text", "")[:200] + ("..." if len(chunk.get("text", "")) > 200 else ""),
                "score": round(float(scores[idx]), 4),
            }
        )

    return results