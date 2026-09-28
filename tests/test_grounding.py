"""Tests for the grounding check module.

Verifies that answers are grounded in the retrieved policy chunks:
- Every cited chunk ID must be in the retrieved set
- Every numeric amount in the answer must appear in a cited chunk
"""

from __future__ import annotations

import pytest

from backend.qa import _grounding_cited_chunks, _grounding_numbers_in_chunks


class TestGrounding:
    """Test grounding checks."""

    def test_cited_chunks_all_in_retrieved(self):
        """All cited chunks should be in the retrieved set."""
        citations = ["A1P1C1", "A1P1C2"]
        retrieved = ["A1P1C1", "A1P1C2", "A1P1C3"]
        result = _grounding_cited_chunks(citations, retrieved)
        assert result is True

    def test_cited_chunk_not_in_retrieved(self):
        """A cited chunk not in retrieved should fail."""
        citations = ["A1P1C1", "A1P1C99"]
        retrieved = ["A1P1C1"]
        result = _grounding_cited_chunks(citations, retrieved)
        assert result is False

    def test_no_citations_trivially_pass(self):
        """No citations should trivially pass."""
        result = _grounding_cited_chunks([], ["A1P1C1"])
        assert result is True

    def test_numbers_in_chunks(self):
        """Numbers in answer should appear in cited chunks."""
        answer = "The room rent cap is Rs. 4,000 per day."
        citations = ["A1P1C1"]
        all_chunks = {
            "test_A": [
                {
                    "chunk_id": "A1P1C1",
                    "text": "Room rent is capped at Rs. 4000 per day for general ward.",
                }
            ]
        }
        result = _grounding_numbers_in_chunks(answer, citations, all_chunks)
        assert result is True

    def test_numbers_not_in_chunks(self):
        """Numbers not in cited chunks should fail."""
        answer = "The deductible is Rs. 15,000 per year."
        citations = ["A1P1C1"]
        all_chunks = {
            "test_A": [
                {
                    "chunk_id": "A1P1C1",
                    "text": "Room rent is capped at Rs. 4000 per day for general ward.",
                }
            ]
        }
        result = _grounding_numbers_in_chunks(answer, citations, all_chunks)
        assert result is False

    def test_no_numbers_trivially_pass(self):
        """No numbers in answer should trivially pass."""
        answer = "What is covered under this policy?"
        citations = ["A1P1C1"]
        all_chunks = {
            "test_A": [
                {
                    "chunk_id": "A1P1C1",
                    "text": "Room rent is capped at Rs. 4000 per day for general ward.",
                }
            ]
        }
        result = _grounding_numbers_in_chunks(answer, citations, all_chunks)
        assert result is True