"""Pydantic models for the PolicyLens API and internal data structures."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload / document state
# ---------------------------------------------------------------------------

class ChunkBase(BaseModel):
    """A single extracted clause/paragraph chunk."""

    chunk_id: str = Field(description='Unique chunk identifier, e.g. "A1P3"')
    doc_id: str = Field(description="Document identifier")
    page: int = Field(description="Page number (1-indexed)")
    section: Optional[str] = Field(
        default=None, description="Detected section heading, if any"
    )
    text: str = Field(description="Full chunk text")
    bboxes: List[List[float]] = Field(
        default_factory=list,
        description="Word bounding boxes [[x0, top, x1, bottom], ...]",
    )
    page_width: float = Field(description="PDF page width in points")
    page_height: float = Field(description="PDF page height in points")


class DocumentUploadResponse(BaseModel):
    """Response from POST /api/upload or POST /api/sample/{name}."""

    doc_id: str
    filename: str
    pages: int
    chunks: int


# ---------------------------------------------------------------------------
# QA / answering output
# ---------------------------------------------------------------------------

class QAAnswer(BaseModel):
    """Structured answer from the QA module."""

    answerable: bool = Field(description="Whether the question can be answered from the policy")
    answer: str = Field(default="", description="The answer text, if answerable")
    citations: List[str] = Field(
        default_factory=list, description="Chunk IDs cited in the answer"
    )


class QAResponse(BaseModel):
    """Response from POST /api/ask."""

    status: str = Field(description="answered | insufficient")
    answer: str = Field(default="", description="The answer text, if answered")
    reason: str = Field(
        default="", description="Human-readable reason, if insufficient"
    )
    evidence_strength: float = Field(
        ge=0.0, le=1.0, description="Derived from retrieval similarity"
    )
    citations: List[str] = Field(
        default_factory=list, description="Chunk IDs cited in the answer"
    )
    nearest: List[Dict[str, Any]] = Field(
        default_factory=list, description="Nearest chunks when insufficient"
    )
    trace_id: str = Field(description="Unique trace identifier for this question")


# ---------------------------------------------------------------------------
# API request models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request body for POST /api/ask."""

    doc_id: str = Field(description="The document to query")
    question: str = Field(description="The user's question")
    history: List[str] = Field(
        default_factory=list, description="Last N turns as context string[]"
    )


class UploadRequest(BaseModel):
    """Request body for POST /api/upload (multipart form)."""

    file: bytes = Field(description="PDF file bytes")


class SampleRequest(BaseModel):
    """Request body for POST /api/sample/{name}."""

    name: str = Field(description="Sample policy name: A, B, or D")


# ---------------------------------------------------------------------------
# Trace entry (JSONL)
# ---------------------------------------------------------------------------

class TraceEntry(BaseModel):
    """One JSONL line appended per question asked."""

    trace_id: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    question: str
    doc_id: str
    retrieved_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs returned by retrieval"
    )
    retrieved_scores: List[float] = Field(
        default_factory=list, description="Similarity scores for retrieved chunks"
    )
    gate_decision: str = Field(
        description="answered | insufficient"
    )
    grounding_pass: bool = Field(
        default=False, description="Whether grounding checks passed"
    )
    latency_ms: float = Field(description="Time from request to response (ms)")


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response from GET /api/health."""

    ok: bool = True
    llm_provider: str = "none"


# ---------------------------------------------------------------------------
# Page image serving
# ---------------------------------------------------------------------------

class PageImageResponse(BaseModel):
    """Placeholder for page image binary response."""

    pass


# ---------------------------------------------------------------------------
# Trace drawer entry (frontend rendering shape)
# ---------------------------------------------------------------------------

class TraceDrawerEntry(BaseModel):
    """Shape rendered in the trace drawer UI."""

    question: str
    retrieved_chunks: List[Dict[str, Any]]
    gate_decision: str
    grounding_results: List[Dict[str, Any]]
    latency_ms: float