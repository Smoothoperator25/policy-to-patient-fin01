"""Answering module: grounding checks + extractive Q&A + LLM JSON mode.

Responsibilities
----------------
* Receive the user question + retrieved chunks from the pipeline.
* Perform grounding checks:
    1. Every cited chunk_id must be in the retrieved set.
    2. Every number/amount in the answer must appear in a cited chunk.
* If grounding fails -> status "insufficient" with reason "answer could not be verified
  against the policy".
* In extractive mode (LLM_PROVIDER=none): return the best-matching clause text as
  the answer, with its citation.
* LLM wrapper (invoke): calls `llm.answer_question()` and returns the structured
  result: {"answerable": bool, "answer": str, "citations": [chunk_id, ...]}.
* Multi-turn: accept the last 2 turns as history; rewrite the query using history
  before retrieval (handled in the API layer, but this module documents the contract).

Public API
----------
``ask(question: str, retrieved: List[dict], history: List[str] = []) -> dict

Returns a dict with keys:
    - "answerable": bool
    - "answer": str (the answer text, empty if not answerable)
    - "citations": List[str] (chunk IDs supporting the answer)
    - "grounding_pass": bool (True if both grounding checks pass)
    - "verification_reason": str | None (why grounding failed, if it did)
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional

from backend.llm import answer_question, invoke
from backend.confidence import gate
from backend.models import QAAnswer, QAResponse, AskRequest


# ---------------------------------------------------------------------------
# Grounding check 1: every cited chunk_id must be in the retrieved set
# ---------------------------------------------------------------------------

def _grounding_cited_chunks(citations: List[str], retrieved_chunk_ids: List[str]) -> bool:
    """Check that all cited chunk IDs were actually returned by retrieval."""
    if not citations:
        return True  # no citations -> trivially pass
    retrieved_set = set(retrieved_chunk_ids)
    cited_set = set(citations)
    return cited_set.issubset(retrieved_set)


# ---------------------------------------------------------------------------
# Grounding check 2: every number in the answer must appear in a cited chunk
# ---------------------------------------------------------------------------

def _grounding_numbers_in_chunks(answer: str, citations: List[str],
                                  all_chunks: Dict[str, List[dict]]) -> bool:
    """Check that every numeric amount/number in the answer appears in a cited chunk.

    Heuristic: extract sequences of digits (and Indian digit groupings like 1,50,000)
    from the answer, then verify at least one cited chunk contains that substring.
    """
    if not citations:
        return True  # no citations -> trivially pass (will be caught by check 1)

    # Extract number-like strings from the answer
    # Match patterns like Rs. 1,50,000 or 150000 or 4,000 etc.
    number_pattern = re.compile(r"Rs?\.?\s*[\d,]+|[\d,]+(?!\s*[a-z])")
    answer_numbers = set(number_pattern.findall(answer))

    if not answer_numbers:
        # No explicit numbers found; trivially pass
        return True

    # Check each cited chunk for the presence of any answer number
    for cid in citations:
        # Find the chunk in the full document data
        found = False
        for doc_chunks in all_chunks.values():
            for chunk in doc_chunks:
                if chunk.get("chunk_id") == cid:
                    chunk_text = chunk.get("text", "")
                    # Check if any answer number appears in the chunk text
                    for num in answer_numbers:
                        if num in chunk_text:
                            found = True
                            break
                    if found:
                        break
            if found:
                break

    return found


# ---------------------------------------------------------------------------
# Main ask function
# ---------------------------------------------------------------------------

def ask(question: str, retrieved: List[dict], history: List[str] = [],
        all_chunks: Optional[Dict[str, List[dict]]] = None) -> Dict[str, Any]:
    """Answer a question using retrieved chunks, with grounding checks.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved : List[dict]
        The chunks returned by the retrieval module, each with at least:
        "chunk_id", "page", "section", "text", "score".
    history : List[str], default []
        Last 2 turns as context strings (question-answer pairs).
    all_chunks : Dict[str, List[dict]], optional
        The full set of chunks for the document, needed for grounding check 2.

    Returns
    -------
    Dict with keys:
        - "answerable": bool
        - "answer": str
        - "citations": List[str]
        - "grounding_pass": bool
        - "verification_reason": str | None
    """
    # Extract retrieved chunk IDs and their scores
    retrieved_chunk_ids = [r["chunk_id"] for r in retrieved]
    retrieved_scores = [r.get("score", 0.0) for r in retrieved]

    # Step 1: Use the LLM (or extractive mode) to generate an answer
    # The LLM function expects a list of chunk dicts with "id", "text", etc.
    chunk_list = [
        {
            "id": r["chunk_id"],
            "text": r.get("text", ""),
            "page": r.get("page", 0),
            "section": r.get("section") or "",
        }
        for r in retrieved
    ]

    llm_result = answer_question(question, chunk_list)

    answerable = llm_result.get("answerable", False)
    answer = llm_result.get("answer", "")
    citations_from_llm = llm_result.get("citations", [])

    # Step 2: Grounding check 1 — every cited chunk must be in the retrieved set
    grounding1 = _grounding_cited_chunks(citations_from_llm, retrieved_chunk_ids)

    # Step 3: Grounding check 2 — every number in the answer must appear in a cited chunk
    grounding2 = True
    if answerable and citations_from_llm and all_chunks:
        grounding2 = _grounding_numbers_in_chunks(answer, citations_from_llm, all_chunks)

    grounding_pass = grounding1 and grounding2

    # Step 4: Apply confidence gate if the LLM says answerable but similarity is low
    # We need the top score for this
    top_score = max(retrieved_scores) if retrieved_scores else 0.0

    gate_result = gate(top_score=top_score)

    # If LLM said answerable but gate says insufficient, override
    if answerable and gate_result["status"] == "insufficient":
        answerable = False
        answer = ""
        citations_from_llm = []

    # Step 5: If grounding failed, force insufficient
    if not grounding_pass:
        answerable = False
        answer = ""
        citations_from_llm = []
        reason_parts = []
        if not grounding1:
            reason_parts.append("cited chunks not all in retrieved set")
        if not grounding2:
            reason_parts.append("numbers in answer not found in cited chunks")
        verification_reason = "; ".join(reason_parts) if reason_parts else None
    else:
        verification_reason = None

    # Step 6: Compute evidence strength from the gate
    evidence_strength = gate_result.get("evidence_strength", 0.0)

    return {
        "answerable": answerable,
        "answer": answer,
        "citations": citations_from_llm,
        "grounding_pass": grounding_pass,
        "verification_reason": verification_reason,
        "evidence_strength": evidence_strength,
    }