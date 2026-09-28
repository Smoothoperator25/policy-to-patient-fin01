# PolicyLens — HackMatrix 5.0 | FIN-01

**Problem:** Patients and clinicians cannot quickly determine what an insurance policy covers, leading to surprise bills and care delays. Policies are long, dense, and filed as PDFs — there is no intelligent way to query them.

**Solution:** PolicyLens: upload an insurance policy PDF, ask coverage questions in plain language, get answers with **page-level citations**, and get an honest "insufficient information" refusal instead of a guess when the policy doesn't support an answer.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [UI/UX Design](#ui-ux-design)
4. [API Contract](#api-contract)
5. [Synthetic Sample Policies](#synthetic-sample-policies)
6. [Built vs Planned](#built-vs-planned)
7. [Team](#team)

---

## Quick Start

Follow these steps to run PolicyLens locally:

```bash
# 1. Clone & enter directory
git clone <repo-url>
cd policy-to-patient-fin01

# 2. Create and activate venv
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (LLM_PROVIDER defaults to "none" for demo)

# 5. Generate sample PDFs
python scripts/make_sample_policies.py

# 6. Start the server
uvicorn backend.app:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python 3.11)                │
│  ├─ ingest.py   : pdfplumber extraction, per-page text + boxes │
│  ├─ chunking.py : clause-aware chunking (60-220 words)          │
│  ├─ retrieval.py: FAISS IndexFlatIP, dense + section lookup     │
│  ├─ confidence.py: threshold gate → "answered" | "insufficient" │
│  ├─ qa.py       : extractive Q&A or LLM JSON output              │
│  ├─ llm.py      : pluggable LLM interface (gemini | anthropic | none)│
│  └─ models.py   : pydantic API contracts + trace schema          │
└───────────────────────┬───────────────────────────────────────┘
                        │  JSONL trace, JSON API, static UI
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vanilla JS Frontend (ES Modules)             │
│  ├─ app.js      : page state, routing, keyboard shortcuts       │
│  ├─ api.js      : fetch API client (upload, ask, trace)        │
│  ├─ viewer.js   : PDF page rendering, highlight sweep animation │
│  ├─ styles.css  : Fraunces/IBM Plex Mono typography, color     │
│  │              tokens, no framework, no gradients               │
│  └─ index.html  : dropzone, conversation, viewer layout         │
└─────────────────────────────────────────────────────────────────┘
```

**Key design choices:**

- **Extractive mode (LLM_PROVIDER=none):** No API key needed. The backend returns the best-matching clause text with its citation. The demo works completely offline.
- **FAISS IndexFlatIP:** In-memory indexes per `doc_id`. Dense retrieval only (k=4) plus exact section/clause force-include.
- **Confidence gate:** If top similarity < threshold (default 0.35), status = "insufficient". Evidence strength mapped from similarity between threshold and ~0.8, labeled "derived from retrieval similarity, not a probability of correctness."
- **Grounding checks:** Every cited chunk_id must be in the retrieved set; every number in the answer must appear in a cited chunk. Failure → "insufficient" with reason "answer could not be verified against the policy."
- **Trace log:** One JSONL line per question (timestamp, question, retrieved chunk_ids + scores, gate decision, grounding result, latency). Expose last 20 via `GET /api/trace`.

---

## UI/UX Design

**Concept: "The Reading Room."** A beautifully typeset legal document annotated with a highlighter, not a SaaS dashboard.

### Design Tokens (CSS variables)

| Variable | Value | Usage |
|----------|-------|-------|
| `--paper` | `#F4EFE6` | Page background |
| `--paper-2` | `#EAE3D4` | Panels |
| `--rule` | `#D9D0BF` | Hairlines |
| `--ink` | `#1B1A17` | Text |
| `--ink-soft` | `#5B574E` | Secondary text |
| `--hl` | `#FFE066` | Highlighter behind cited text and chips |
| `--stamp` | `#C4321F` | Refusals and errors |

### Typography

- **Headings:** Fraunces (Google Font) — serif
- **Body:** Instrument Sans (Google Font) — fallbacks: Georgia, system-ui
- **Citations, numbers, labels:** IBM Plex Mono (Google Font) — fallbacks: `ui-monospace`, `system-ui`
- **Scale:** 13 / 15 / 17 / 22 / 32
- **Body line-height:** 1.6
- **Answer text max width:** 62ch

### Forbidden (anti-patterns)

- Purple/blue gradients, glassmorphism, bento cards, chat bubbles
- Inter/Roboto, emoji, sparkle or "AI-powered" badges
- Centered marketing hero, rounded-2xl everything, dark-mode default
- Stock illustrations, lorem ipsum, "Oops!" copy

### Layout (desktop >= 1100px)

- **Top bar:** wordmark "PolicyLens" (Fraunces, letter-spaced), current document name, "p. 7 / 24" in mono, "SYNTHETIC DEMO DATA" tag when a sample is loaded, Trace button
- **Left pane (46%):** conversation, composer pinned at bottom
- **Right pane (54%):** source document viewer on `--paper-2`, PDF page centered like a sheet of paper, page navigator (prev/next, jump), fit-width

### Screens and States

1. **Empty:** Large dashed-ink dropzone with headline "File a policy." Subtext: "PDF with selectable text works best." Three chips: "Try Sample A / B / D"
2. **Parsing:** Indeterminate ruled-line loader + "Reading the policy..." Real counts shown when done: "24 pages, 187 clauses indexed."
3. **Ready:** First page appears in viewer. 4 suggested-question chips: "What is covered for hospitalization?", "What is excluded?", "Are there waiting periods?", "What is the room rent limit?"
4. **Answered:** Numbered memo entry — mono "No. 03" label, question in Fraunces italic, answer in body text, citation row, evidence meter
5. **Refusal:** Rectangular double-border stamp in `--stamp`, mono uppercase "INSUFFICIENT INFORMATION", rotated -2.5deg. Plain-language reasoning + "What would help" line. Closest text found (not an answer) listing nearest clauses with citation chips. Never gray "I don't know" text.
6. **Error:** Stamp-style error box with specific copy. No exclamation marks.
7. **Trace drawer:** Slides in from the right, toggled by Trace button or T key. Shows retrieved chunks with similarity scores, gate decision vs threshold, grounding check results, latency.
8. **(STRETCH) Estimate cost tab:** Receipt-style layout with dotted leaders, Indian digit grouping, banner "PREVIEW - SAMPLE DATA, NOT CONNECTED".

### Motions (exactly three)

- Highlight sweep: citation chip clicked → rectangles sweep in left-to-right over 600ms
- Stamp land: refusal stamp with 220ms scale-down (1.15 to 1) plus fade
- 180ms message fade-up

**Honor prefers-reduced-motion:** disable all three animations.

### Accessibility

- WCAG AA contrast (yellow only as background behind ink text)
- 2px ink focus ring
- Chips are real buttons
- Stamp is real text with `role="status"` `aria-live="polite"`
- Page image has alt "Policy page N of M"
- Full keyboard support

### Keyboard

- `/` focuses the composer
- Enter sends, Shift+Enter newline
- T toggles Trace
- `[` and `]` flip pages
- Esc closes the drawer

### Microcopy

- Composer placeholder: "Ask about coverage, limits, exclusions..."
- Amounts formatted like `Rs. 1,50,000`

### Fallback

- If page-image rendering fails, viewer.js falls back to the text endpoint and uses `<mark>` highlighting. The demo must never break.

---

## API Contract

| Endpoint | Request | Response |
|----------|---------|----------|
| `POST /api/upload` | multipart PDF | `{doc_id, filename, pages, chunks}` |
| `POST /api/sample/{name}` | — | Same as upload, for A, B, D |
| `POST /api/ask` | `{doc_id, question, history[]}` | `{status: "answered"|"insufficient", answer, reason, evidence_strength, citations:[{chunk_id, page, section, quote, bboxes, page_width, page_height}], nearest:[same shape, only when insufficient], trace_id}` |
| `GET /api/page/{doc_id}/{n}.png` | — | Page image at ~110 dpi (cache in page_cache/) |
| `GET /api/page/{doc_id}/{n}/text` | — | Plain page text (fallback viewer) |
| `GET /api/trace` | — | Last 20 trace entries |
| `GET /api/health` | — | `{ok:true, llm_provider}` |

---

## Synthetic Sample Policies

Three short realistic synthetic health-policy PDFs (5-8 pages each) generated with `reportlab`, each footer-labeled `"SYNTHETIC SAMPLE - NOT A REAL POLICY"`:

- **Policy A:** Covers hospitalization, room rent cap, co-pay, exclusions, waiting periods, deductible.
- **Policy B:** Same as A, plus an eligibility clause (coverage for a named procedure applies to ages 18-65).
- **Policy D:** Deliberately silent on one sub-limit (e.g., ICU charges) so the refusal path is demonstrable.

Data file: `data/treatment_costs.csv` — 10+ procedures with min/max cost, category, length of stay, tier1/tier2/tier3 multipliers. Header comment: `"SYNTHETIC REFERENCE DATA"`.

**Important:** These are sample data only. The cost-estimate tab (STRETCH) labels output as `PREVIEW - SAMPLE DATA, NOT CONNECTED`. Never present mock statistics as real output.

---

## Built vs Planned

### BUILT (Round 1 / MUST)

- PDF upload and per-page text extraction with pdfplumber
- Clause-aware chunking (60-220 words, never merge across pages)
- Embeddings + FAISS retrieval (dense top-k=4 + section/clause force-include)
- Confidence gate with threshold (default 0.35, calibrate via script)
- Cited Q&A in extractive mode (best-matching clause text + citation)
- Grounding checks: every cited chunk_id in retrieved set; every number in answer appears in cited chunk
- Multi-turn: last 2 turns as history, query rewrite before retrieval
- Privacy: uploads temp dir, deleted after 1 hour and on shutdown
- Trace JSONL log, last 20 entries via API
- Full UI: Reading Room layout, all 7 required screens/states
- Three synthetic sample policies (A, B, D)
- Tests: chunking, confidence, grounding
- README, architecture diagram, tech stack, setup instructions

### PLANNED (future phases / STRETCH)

- Rule-based clause-type tagger
- Cost-estimate preview tab (receipt-style with Indian digit grouping)
- Eligibility check (age-based clause evaluation)
- Validator suite for grounding checks
- OCR support (planned, not built)
- ML cost-regression model
- Clause classifier (rule-based → learned)
- Multi-agent orchestration

---

## Privacy Notes

- Uploads go to a temp dir and are deleted after 1 hour and on shutdown
- Never log full document text — only retrieved chunks are sent to an external LLM
- Trace entries contain question text and chunk_ids + scores — no full document text
- `.env` is gitignored; never commit API keys or secrets

---

## Tech Stack (fixed, do not substitute)

- **Backend:** Python 3.11, FastAPI, uvicorn, pdfplumber, sentence-transformers (all-MiniLM-L6-v2), faiss-cpu, numpy, pydantic, python-dotenv, pytest, reportlab (sample PDFs only)
- **Frontend:** plain HTML + CSS + vanilla JS (ES modules), served by FastAPI as static files. No build step, no framework, no CSS library
- **LLM:** pluggable via `LLM_PROVIDER` env var = gemini \| anthropic \| none, in `backend/llm.py` behind one function interface. Default `"none"` = extractive mode
- **Configuration via `.env` (.env.example committed, `.env` gitignored)**

---

## Setup Checklist

- [ ] `python -m venv .venv && . .venv\Scripts\activate`
- [ ] `pip install -r requirements.txt`
- [ ] `cp .env.example .env`
- [ ] `python scripts/make_sample_policies.py`
- [ ] `uvicorn backend.app:app --reload`
- [ ] Open http://localhost:8000

---

## Team

**Tech Strikas** — HackMatrix 5.0 | FIN-01: Policy-to-Patient

---