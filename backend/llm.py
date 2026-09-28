"""Pluggable LLM interface: gemini | anthropic | none (extractive mode).

The backend operates in extractive mode by default (LLM_PROVIDER=none),
which means no API key and no internet are required. The LLM function
returns strict JSON {"answerable": bool, "answer": str, "citations": [chunk_id,...]}
or, in extractive mode, the best-matching clause text as the answer.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Embedding helper (re-use sentence-transformers all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

from sentence_transformers import SentenceTransformer

_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazily load the all-MiniLM-L6-v2 sentence transformer."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# Core LLM function
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    context_chunks: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Answer a question using only the provided context chunks.

    Parameters
    ----------
    question: str
        The user's question.
    context_chunks: List[Dict[str, Any]]
        Each dict must contain at least:
        - "id": chunk identifier str
        - "doc_id": document identifier str
        - "page": page number int
        - "section": section heading str | None
        - "text": clause text str
    doc_context: Optional[Dict[str, Any]]
        Optional broader document context (not used in extractive mode).

    Returns
    -------
    Dict with keys:
        - "answerable": bool
        - "answer": str (the answer, or empty if not answerable)
        - "citations": List[str] (chunk IDs supporting the answer)
    """
    provider = os.getenv("LLM_PROVIDER", "none")

    # ── Extractive mode (NO API KEY, NO INTERNET) ──────────────────────
    if provider == "none" or provider is None:
        return _extractive_answer(question, context_chunks)

    # ── Gemini provider ────────────────────────────────────────────────
    if provider == "gemini":
        return _gemini_answer(question, context_chunks, doc_context)

    # ── Anthropic provider ─────────────────────────────────────────────
    if provider == "anthropic":
        return _anthropic_answer(question, context_chunks, doc_context)

    # ── Unknown provider fallback ───────────────────────────────────────
    return _extractive_answer(question, context_chunks)


def _extractive_answer(
    question: str, context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extractive mode: return the best-matching clause as the answer.

    Scoring is based on cosine similarity between the question embedding
    and each chunk's embedding. The top-scoring chunk is returned with
    its full text as the answer, along with its citation.

    If no chunk achieves a meaningful similarity, the question is deemed
    unanswerable and answerable=false.
    """
    if not context_chunks:
        return {"answerable": False, "answer": "", "citations": []}

    # Build question + chunk texts for batch encoding
    q = question
    texts = [chunk["text"] for chunk in context_chunks]

    # Embed using all-MiniLM-L6-v2
    model = _get_embedding_model()
    embeddings = model.encode([q] + texts, normalize_embeddings=True)

    q_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    # Cosine similarity (vectors are normalized, so just dot product)
    sims = np.dot(chunk_embs, q_emb)

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    # If the best similarity is very low, treat as unanswerable
    # Threshold calibrated per-confidence-gate logic; here we use a soft floor
    if best_score < 0.15:
        return {"answerable": False, "answer": "", "citations": []}

    best_chunk = context_chunks[best_idx]
    answer_text = best_chunk["text"].strip()[:500]  # truncate for readability
    citations = [best_chunk["id"]]

    return {
        "answerable": True,
        "answer": answer_text,
        "citations": citations,
    }


def _gemini_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Gemini API provider.

    Plugged in behind the same interface. Calls the Google Gemini API
    with the retrieved chunks as grounded context. Returns the same
    schema as _extractive_answer.
    """
    try:
        from google import genai
    except ImportError:
        return {
            "answerable": False,
            "answer": "",
            "citations": [],
            "_error": "google-genai not installed",
        }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "answerable": False,
            "answer": "",
            "citations": [],
            "_error": "GEMINI_API_KEY not set",
        }

    # Build the prompt from retrieved chunks
    context_text = "\n\n".join(
        f"[Clause {c.get('id', '?')} p.{c.get('page', '?')} {c.get('section', '')}]: "
        f"{c['text']}"
        for c in context_chunks
    )

    system_prompt = """You are PolicyLens, an insurance policy assistant.
Answer the user's question using ONLY the provided clause text.
Every claim must cite chunk IDs.
Quote numbers and amounts verbatim.
If the clauses don't answer it, return {"answerable": false}.
Do not give medical or legal advice.
Return strict JSON only: {"answerable": bool, "answer": str, "citations": [chunk_id, ...]}."""

    user_prompt = f"""Context clauses:
{context_text}

Question: {question}

Answer (strict JSON):"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_prompt},
        )

        # Parse the response text as JSON
        text = response.text.strip()
        try:
            result = json.loads(text)
            # Validate required keys
            if (
                isinstance(result, dict)
                and "answerable" in result
                and "answer" in result
                and "citations" in result
            ):
                return result
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract answerability from text
        if "answerable" in text.lower() and "false" in text.lower():
            return {"answerable": False, "answer": "", "citations": []}
        return {"answerable": True, "answer": text, "citations": []}

    except Exception as e:
        # Gemini failed — fall back to extractive mode
        print(f"[LLM] Gemini error, falling back extractive: {e}")
        return _extractive_answer(question, context_chunks)


def _anthropic_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
    doc_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Anthropic API provider.

    Plugged in behind the same interface. Calls the Anthropic Claude API
    with the retrieved chunks as grounded context.
    """
    try:
        import anthropic
    except ImportError:
        return {
            "answerable": False,
            "answer": "",
            "citations": [],
            "_error": "anthropic not installed",
        }

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "answerable": False,
            "answer": "",
            "citations": [],
            "_error": "ANTHROPIC_API_KEY not set",
        }

    # Build context text
    context_text = "\n\n".join(
        f"[Clause {c.get('id', '?')} p.{c.get('page', '?')} {c.get('section', '')}:]"
        f" {c['text']}"
        for c in context_chunks
    )

    system_prompt = """You are PolicyLens, an insurance policy assistant.
Answer the user's question using ONLY the provided clause text.
Every claim must cite chunk IDs.
Quote numbers and amounts verbatim.
If the clauses don't answer it, return {"answerable": false}.
Do not give medical or legal advice.
Return strict JSON only: {"answerable": bool, "answer": str, "citations": [chunk_id, ...]}."""

    user_prompt = f"""Context clauses:
{context_text}

Question: {question}

Answer (strict JSON):"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.generate(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.text.strip()
        try:
            result = json.loads(text)
            if (
                isinstance(result, dict)
                and "answerable" in result
                and "answer" in result
                and "citations" in result
            ):
                return result
        except json.JSONDecodeError:
            pass

        if "answerable" in text.lower() and "false" in text.lower():
            return {"answerable": False, "answer": "", "citations": []}
        return {"answerable": True, "answer": text, "citations": []}

    except Exception as e:
        print(f"[LLM] Anthropic error, falling back extractive: {e}")
        return _extractive_answer(question, context_chunks)


# ---------------------------------------------------------------------------
# Convenience wrapper used by the QA module
# ---------------------------------------------------------------------------

def invoke(
    question: str,
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Answer a question using only the retrieved chunks.

    Parameters
    ----------
    question: str
        The user question.
    retrieved: List[Dict[str, Any]]
        Chunks returned by the retrieval module, each with at least
        "id", "text", "page", "section" keys.

    Returns
    -------
    Dict with keys "answerable", "answer", "citations".
    """
    return answer_question(question, retrieved)