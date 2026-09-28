"""FastAPI backend for PolicyLens.

Endpoints (all JSON unless otherwise noted):
----------------------
GET   /api/health                -> {ok:true, llm_provider}
GET   /api/trace                 -> last 20 JSONL trace entries
GET   /api/page/{doc_id}/{n}.png -> page image (PNG)
GET   /api/page/{doc_id}/{n}/text -> plain page text
POST  /api/upload                -> multipart PDF -> {doc_id, filename, pages, chunks}
POST  /api/sample/{name}         -> same as upload, for A, B, D
POST  /api/ask                   -> {doc_id, question, history[]} -> answer
POST  /api/upload                -> multipart PDF upload

Background
----------
* Uploads land in a temp dir (set by ``PDF_UPLOAD_DIR`` env var, default "uploads").
* After 1 hour (or on process shutdown) temp files are deleted.
* Trace entries are appended as JSONL to ``.cache/trace.jsonl``.
* Only the retrieved chunks (never full document text) are sent to an external LLM.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.ingest import extract_chunks
from backend.retrieval import retrieve
from backend.confidence import gate, _get_threshold
from backend.qa import ask
from backend.models import (
    ChunkBase,
    QAResponse,
    AskRequest,
    QAAnswer,
    DocumentUploadResponse,
    TraceEntry,
    HealthResponse,
    PageImageResponse,
)

# ---------------------------------------------------------------------------
# Directory configuration
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(os.getenv("PDF_UPLOAD_DIR", "uploads"))
TRACE_DIR = Path(os.getenv("TRACE_DIR", ".cache"))
TRACE_FILE = TRACE_DIR / "trace.jsonl"
MAX_PAGES = int(os.getenv("MAX_PAGES", "50"))
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory document store: doc_id -> List[ChunkBase]
# ---------------------------------------------------------------------------

doc_store: Dict[str, List[ChunkBase]] = {}

# Trace lock for thread-safe writing
_trace_lock = threading.Lock()

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(title="PolicyLens", version="0.1.0")

# Mount static frontend (served from the repo root's ../frontend/ relative to backend/)
# We'll serve from the parent directory's "frontend" folder at runtime.
# For now, mount a basic static dir if it exists.
_frontent_path = Path(__file__).parent.parent / "frontend"
if _frontent_path.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontent_path)), name="frontend")


# ---------------------------------------------------------------------------
# Helper: append a trace entry as JSONL
# ---------------------------------------------------------------------------

def _append_trace(entry: TraceEntry) -> None:
    """Append one JSONL line to the trace file."""
    with _trace_lock:
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Helper: clean old uploads (>1 hour) and old trace entries
# ---------------------------------------------------------------------------

def _cleanup_old_files() -> None:
    """Remove uploads older than 1 hour and prune trace to last 1000 lines."""
    now = datetime.now()
    # Old uploads
    if UPLOAD_DIR.is_dir():
        for p in UPLOAD_DIR.iterdir():
            if p.is_file():
                try:
                    mtime = p.stat().st_mtime
                    if (now - datetime.fromtimestamp(mtime)).total_seconds() > 3600:
                        p.unlink()
                except (OSError, ValueError):
                    pass

    # Prune trace file to last 1000 lines
    if TRACE_FILE.is_file():
        try:
            with open(TRACE_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > 1000:
                with open(TRACE_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines[-1000:])
        except (OSError, UnicodeDecodeError):
            pass


# Schedule periodic cleanup (every 6 hours via a simple thread, or on shutdown)
# For simplicity, we'll run cleanup on each request's background


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return the health of the service and the LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "none")
    return HealthResponse(ok=True, llm_provider=provider)


@app.get("/api/trace", response_model=List[TraceEntry])
async def get_trace() -> List[TraceEntry]:
    """Return the last 20 trace entries (JSONL)."""
    if not TRACE_FILE.is_file():
        return []
    try:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Return last 20 entries
        entries: List[TraceEntry] = []
        for line in lines[-20:]:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(TraceEntry(**data))
            except json.JSONDecodeError:
                continue
        return entries
    except Exception:
        return []


@app.get("/api/page/{doc_id}/{n}/text")
async def page_text(doc_id: str, n: int) -> Dict[str, Any]:
    """Return plain text for page n of a document.

    The PDF is stored under the uploads dir as {doc_id}.pdf.
    """
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    if not pdf_path.is_file():
        return {"error": "Document not found or PDF not available."}

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if n < 1 or n > len(pdf.pages):
                return {"error": f"Page {n} out of range (1-{len(pdf.pages)})"}
            page = pdf.pages[n - 1]
            text = page.text or ""
        return {"page": n, "text": text}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/page/{doc_id}/{n}.png")
async def page_png(doc_id: str, n: int) -> FileResponse:
    """Return a page image (PNG) at ~110 dpi.

    The PDF is stored under the uploads dir as {doc_id}.pdf.
    Images are cached under page_cache/.
    """
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    cache_dir = Path("page_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{doc_id}_p{n}.png"

    # If cached, serve from cache
    if cache_path.is_file():
        return FileResponse(str(cache_path), media_type="image/png")

    if not pdf_path.is_file():
        return JSONResponse(status_code=404, content={"error": "Document not found"})

    if n < 1:
        return JSONResponse(status_code=400, content={"error": "Page number must be >= 1"})

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if n > len(pdf.pages):
                return JSONResponse(status_code=400, content={"error": f"Page {n} out of range"})
            page = pdf.pages[n - 1]

            # Render the page as an image at ~110 dpi
            # pdfplumber uses the page's width/height in points; 110 dpi ~= 110/72 scale
            scale = 110 / 72
            rend = page.to_image(resolution=110 * 72 / 72)  # target 110 dpi
            # Actually let's use the render method
            # pdfplumber page.to_image takes a resolution parameter (DPI)
            img = page.to_image(resolution=110)

            # Save to cache buffer
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            # Write to cache
            with open(cache_path, "wb") as f:
                f.write(buf.read())

            return FileResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload a PDF policy document.

    - Validates file type and size.
    - Saves the temp file under the uploads dir.
    - Extracts per-page text, word bboxes, and clause-aware chunks.
    - Stores the chunks in the in-memory doc_store.
    - Returns the doc_id, page count, and chunk count.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("application/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file type. Expected a PDF."},
        )

    # Read file bytes
    bytes_data = await file.read()

    # Size check
    if len(bytes_data) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=400,
            content={"error": f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)}MB."},
        )

    # Save temp file
    filename = file.filename or "policy.pdf"
    # Sanitize: keep only safe chars
    safe_name = re.sub(r"[^\w\.\-]", "_", filename)
    # Use doc_id based on filename stem
    doc_id = re.sub(r"[^\w]", "_", safe_name.rsplit(".", 1)[0].lower()) or "policy"

    # Avoid overwriting existing doc_ids; append a counter if needed
    original_doc_id = doc_id
    counter = 0
    while doc_id in doc_store:
        counter += 1
        doc_id = f"{original_doc_id}_{counter}"

    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"
    try:
        with open(pdf_path, "wb") as f:
            f.write(bytes_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not save file: {e}"})

    # Extract chunks
    try:
        chunks, error_msg = extract_chunks(str(pdf_path), doc_id)
    except Exception as e:
        # Clean up the saved file on error
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": f"Extraction failed: {e}"})

    # Store in doc_store
    doc_store[doc_id] = chunks

    # Page count
    page_count = len(chunks) // 20 + 1  # rough estimate; actual from PDF
    # Better: re-read page count from the PDF
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_count = len(pdf.pages)
    except Exception:
        pass

    # Cleanup old files after successful upload
    _cleanup_old_files()

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=filename,
        pages=page_count,
        chunks=len(chunks),
    )


@app.post("/api/sample/{name}", response_model=DocumentUploadResponse)
async def sample_policy(name: str, request: Request) -> DocumentUploadResponse:
    """Load one of the three synthetic sample policies (A, B, D).

    Each sample is generated by ``scripts/make_sample_policies.py`` and stored
    under ``data/sample_policies/``.
    """
    # Path to sample PDFs
    samples_dir = Path(__file__).parent.parent / "data" / "sample_policies"
    pdf_path = samples_dir / f"{name}.pdf"

    if not pdf_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"error": f"Sample '{name}' not found. Available: A, B, D."},
        )

    # Read the PDF bytes
    try:
        with open(pdf_path, "rb") as f:
            bytes_data = f.read()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not read sample: {e}"})

    # Use a doc_id based on the sample name
    doc_id = f"sample_{name}"

    # Avoid overwriting
    original_doc_id = doc_id
    counter = 0
    while doc_id in doc_store:
        counter += 1
        doc_id = f"{original_doc_id}_{counter}"

    # Save to uploads dir (temporarily, will be in doc_store)
    pdf_path_out = UPLOAD_DIR / f"{doc_id}.pdf"
    try:
        with open(pdf_path_out, "wb") as f:
            f.write(bytes_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Could not save sample: {e}"})

    # Extract chunks
    try:
        chunks, error_msg = extract_chunks(str(pdf_path_out), doc_id)
    except Exception as e:
        try:
            pdf_path_out.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": f"Extraction failed: {e}"})

    # Store in doc_store
    doc_store[doc_id] = chunks

    # Page count
    page_count = 0
    try:
        with pdfplumber.open(str(pdf_path_out)) as pdf:
            page_count = len(pdf.pages)
    except Exception:
        pass

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=f"{name}.pdf",
        pages=page_count,
        chunks=len(chunks),
    )


@app.post("/api/ask", response_model=QAResponse)
async def ask_question(payload: AskRequest, request: Request) -> QAResponse:
    """Ask a question about a policy document.

    Request body:
        - doc_id: str   (the document to query)
        - question: str (the user's question)
        - history: List[str]  (last 2 turns as context strings)

    Response:
        - status: "answered" | "insufficient"
        - answer: str
        - reason: str (human-readable, when insufficient)
        - evidence_strength: float in [0, 1]
        - citations: List[dict] (each with chunk_id, page, section, quote, bboxes, ...)
        - nearest: List[dict] (only when insufficient, closest chunks with scores)
        - trace_id: str
    """
    start_time = datetime.now()

    doc_id = payload.doc_id
    question = payload.question
    history = payload.history or []

    # Validate document exists
    if doc_id not in doc_store:
        return QAResponse(
            status="insufficient",
            answer="",
            reason="Document not found. Please upload a policy first.",
            evidence_strength=0.0,
            citations=[],
            nearest=[],
            trace_id="",
        )

    chunks = doc_store[doc_id]

    # Build the doc_chunks dict for the retrieval module
    # Group chunks by doc_id (we only have one doc, but the retrieval API expects a dict)
    doc_chunks: Dict[str, List[dict]] = {doc_id: [chunk.model_dump() for chunk in chunks]}

    # Step 1: Retrieve (dense FAISS top-k + section/clause force-include)
    retrieved = retrieve(question, doc_chunks, k=4)

    # Extract chunk IDs and scores from retrieved
    retrieved_chunk_ids = [r["chunk_id"] for r in retrieved]
    retrieved_scores = [r.get("score", 0.0) for r in retrieved]

    # Step 2: QA - generate answer using LLM (or extractive mode)
    qa_result = ask(question, retrieved, history=history, all_chunks=doc_chunks)

    # Step 3: Confidence gate
    top_score = max(retrieved_scores) if retrieved_scores else 0.0
    gate_result = gate(top_score=top_score)

    # Step 4: Build the final response
    trace_id = f"trace_{datetime.now().isoformat().replace(':', '_').replace('.', '_')}"

    # If gate says insufficient, override status
    final_status = gate_result["status"]
    final_answer = qa_result.get("answer", "") if qa_result.get("answerable") else ""
    final_evidence = gate_result["evidence_strength"]
    final_reason = ""

    if final_status == "insufficient":
        final_reason = gate_result.get("reason", "Insufficient information in the policy to answer this question.")
        # Include nearest chunks when insufficient
        nearest_chunks = []
        for r in retrieved:
            nearest_chunks.append(
                {
                    "chunk_id": r["chunk_id"],
                    "page": r.get("page", "?"),
                    "section": r.get("section", "?"),
                    "quote": r.get("text", "")[:160] + ("..." if len(r.get("text", "")) > 160 else ""),
                    "score": r.get("score", 0.0),
                }
            )
    else:
        nearest_chunks = []
        # Build citation rows with full details
        citation_rows = []
        for cit_id in qa_result.get("citations", []):
            # Find the chunk details
            chunk = next(
                (c for c in chunks if c.chunk_id == cit_id),
                None,
            )
            if chunk:
                citation_rows.append(
                    {
                        "chunk_id": cit_id,
                        "page": chunk.page,
                        "section": chunk.section or "?",
                        "quote": chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                        "bboxes": chunk.bboxes,
                        "page_width": chunk.page_width,
                        "page_height": chunk.page_height,
                    }
                )

    # Step 5: Append trace entry
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000
    trace_entry = TraceEntry(
        trace_id=trace_id,
        timestamp=datetime.now().isoformat() + "Z",
        question=question,
        doc_id=doc_id,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_scores=retrieved_scores,
        gate_decision=final_status,
        grounding_pass=qa_result.get("grounding_pass", False),
        latency_ms=latency_ms,
    )
    _append_trace(trace_entry)

    # Step 6: Periodic cleanup
    _cleanup_old_files()

    # Construct the response
    citations_detail = citation_rows if citation_rows else []

    response = QAResponse(
        status=final_status,
        answer=final_answer,
        reason=final_reason,
        evidence_strength=final_evidence,
        citations=citations_detail,
        nearest=nearest_chunks if final_status == "insufficient" else [],
        trace_id=trace_id,
    )

    return response