"""Tests for the confidence gate module.

Verifies that the threshold gate correctly classifies questions as
answered or insufficient based on retrieval similarity scores.
"""

from __future__ import annotations

import pytest

from backend.confidence import gate


class TestConfidenceGate:
    """Test the confidence gate function."""

    def test_below_threshold_insufficient(self):
        """Score below threshold should return 'insufficient'."""
        result = gate(top_score=0.2, threshold=0.35)
        assert result["status"] == "insufficient"
        assert result["evidence_strength"] < 1.0
        assert "reason" in result

    def test_at_threshold_answered(self):
        """Score at threshold should return 'answered'."""
        result = gate(top_score=0.35, threshold=0.35)
        assert result["status"] == "answered"
        assert result["evidence_strength"] == 0.0

    def test_above_threshold_answered(self):
        """Score above threshold should return 'answered'."""
        result = gate(top_score=0.5, threshold=0.35)
        assert result["status"] == "answered"
        assert result["evidence_strength"] > 0.0

    def test_high_score_full_evidence(self):
        """High score should return evidence_strength = 1.0."""
        result = gate(top_score=0.8, threshold=0.35)
        assert result["status"] == "answered"
        assert result["evidence_strength"] == 1.0

    def test_very_high_score_capped(self):
        """Very high score should be capped at 1.0."""
        result = gate(top_score=0.95, threshold=0.35)
        assert result["evidence_strength"] == 1.0

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        result = gate(top_score=0.3, threshold=0.25)
        assert result["status"] == "answered"
        assert result["evidence_strength"] > 0.0