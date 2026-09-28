"""Confidence gate: threshold-based gate that returns "answered" or "insufficient".

Responsibilities
----------------
* Compare the top retrieval similarity score against a configurable threshold
  (default 0.35, stored in the environment variable ``CONFIDENCE_THRESHOLD``).
* If the top score < threshold, gate status = "insufficient".
* Evidence strength is mapped from the similarity between the threshold and ~0.8,
  linearly interpolated so that:
    - similarity == threshold   -> evidence_strength = 0.0
    - similarity >= 0.8         -> evidence_strength = 1.0
* The gate decision and evidence strength are returned so the API can surface
  the "Evidence meter" and the "INSUFFICIENT INFORMATION" stamp.
* The label on the evidence meter is: "Derived from retrieval similarity, not a
  probability of correctness."

Public API
----------
``gate(top_score: float, threshold: float | None = None) -> dict

Returns a dict with:
    - "status": "answered" | "insufficient"
    - "evidence_strength": float in [0, 1]
    - "reason": str (human-readable, only when insufficient)
"""

from __future__ import annotations

import os
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

def _get_threshold() -> float:
    """Read the confidence threshold from the environment.

    Default: 0.35.  The value can be overridden via the ``CONFIDENCE_THRESHOLD``
    environment variable (useful for calibration scripts).
    """
    try:
        return float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    except ValueError:
        return 0.35


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def gate(
    top_score: float,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """Evaluate the confidence gate for a retrieval result.

    Parameters
    ----------
    top_score : float
        The highest similarity score from FAISS retrieval (range [-1, 1]).
        Cosine similarity with IndexFlatIP is in [0, 1] for normalized vectors.
    threshold : float | None
        Override the default threshold. Useful for testing or calibration.

    Returns
    -------
    Dict with keys:
        - "status": "answered" if top_score >= threshold else "insufficient"
        - "evidence_strength": float in [0, 1], mapped from the similarity
          between *top_score* and the interval [threshold, ~0.8].
          - When top_score >= threshold: 0.0 (at threshold) -> 1.0 (at 0.8)
          - When top_score < threshold: mapped to [0, 0.5] to reflect
            lower confidence, since the question is unanswerable.
        - "reason": human-readable explanation (only present when status is
          "insufficient").
    """
    if threshold is None:
        threshold = _get_threshold()

    # Clamp top_score to [0, 1] in case of edge cases
    top_score = max(0.0, min(1.0, top_score))

    if top_score >= threshold:
        # Above threshold — we can answer
        # Map from [threshold, 0.8] -> [0, 1]
        evidence_strength = round(
            min(1.0, (top_score - threshold) / (0.8 - threshold)), 3
        ) if top_score < 0.8 else 1.0
        return {
            "status": "answered",
            "evidence_strength": evidence_strength,
            "reason": "",
        }
    else:
        # Below threshold — insufficient information
        # Map to [0, 0.5] to reflect lower confidence.
        # A score of 0 -> 0.0 evidence, score at threshold -> 0.5 evidence
        evidence_strength = round((top_score / threshold) * 0.5, 3) if threshold > 0 else 0.0

        reason = "The retrieved clauses do not provide sufficient similarity to confidently answer the question. " \
                 "The top retrieved chunk has a similarity score below the confidence threshold."

        return {
            "status": "insufficient",
            "evidence_strength": evidence_strength,
            "reason": reason,
        }