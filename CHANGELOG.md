# Changelog

All notable changes to Genie are documented here.

---

## [v2.0.0] — 2026-02-21 — Hackathon 2: Trust Enforcement Layer

### Overview
Genie transitions from a retrieval prototype to an enterprise-grade trust system.  
Hallucination prevention is now enforced by architecture, not by prompt instruction.

### Added

#### Enforcement Pipeline
- **`gatekeeper.py`** — Dual-enforcement decision engine. Every answer must satisfy two independent conditions before passing: `grounding_score >= threshold` AND `has_valid_citations(answer) == True`. Answers that fail either gate are blocked.
- **`grounding.py`** — Computes cosine similarity between the generated answer and retrieved document chunks using `sentence-transformers`. This is the grounding score displayed in the UI.
- **`citation_validator.py`** — Detects Harvard-style inline citations in LLM responses. Supports both full format `(Author, Year, p. N)` and document-native format `(filename.pdf, n.d.)`.
- **`trace_logger.py`** — Persistent JSONL audit logging. Every query is appended to `genie_trace_log.jsonl` with timestamp, query, answer, grounding score, decision, and sources — regardless of outcome.

#### Decision States (always visible in UI)
- ✅ **PASS** — Answer is grounded and citation-verified. Full Harvard citations shown.
- ⚠️ **SYNTHESIS** — Answer synthesised from document content. Grounding score shown. No direct citation.
- 🚫 **BLOCK** — Answer cannot be verified. SME escalation triggered. Grounding score shown.

#### Intent Detection (FAISS-distance routing)
Replaced brittle keyword matching with distance-threshold routing:
- Distance ≤ 1.2 → factual retrieval path (strict Mistral prompt)
- Distance 1.2–2.0 → comprehension synthesis path (permissive Mistral prompt)
- Distance > 2.0 → out-of-scope refusal

#### Prompt Engineering
- Both Mistral prompts (`call_mistral_api`, `call_mistral_comprehension`) now require inline source citation in responses using `(filename, n.d.)` format, enabling downstream citation validation.

### Changed
- **Confidence score** now derived from grounding cosine similarity (accurate) instead of FAISS L2 distance formula (proxy). Confidence and grounding score are now the same metric, displayed consistently.
- **`show_answer()`** refactored to accept a `decision` parameter (`PASS`/`SYNTHESIS`) instead of a Boolean `show_sources` flag — provenance state is now always disclosed, never silently hidden.
- **`show_sme_escalation()`** now shows a grounding score caption and a 🚫 BLOCK badge, not just a warning.

### Removed
- `COMPREHENSION_KEYWORDS` list — keyword-based intent detection removed entirely.
- `is_comprehension_query()` function — replaced by FAISS distance routing.

### Fixed
- Export JSON now includes `decision` and `grounding_score` fields for audit trail completeness.
- Sources list in JSON export is `[]` for SYNTHESIS decisions (correct — no direct document citation).

### Repository
- Removed test files and `__pycache__` from production repo.
- `.gitignore` updated to exclude `.env`, `genie_trace_log.jsonl`, `.venv/`, type-checker configs.
- `requirements.txt` updated with complete production dependency set.

---

## [v1.0.0] — 2026-02-16 — Hackathon 1: Prototype

### Overview
Genie's initial working prototype. Private document ingestion, vector search, and LLM-powered question answering over uploaded files.

### Added

#### Core Application (`app.py`)
- **File upload** — Accepts PDF, DOCX, and TXT files (up to 3, 500MB each).
- **Text extraction** — PyPDF for PDFs, python-docx for DOCX, UTF-8 decode for TXT.
- **Chunking** — Documents split into 500-character chunks with 50-character overlap for retrieval quality.
- **Embedding** — Chunks embedded using `sentence-transformers/all-MiniLM-L6-v2`.
- **FAISS vector store** — L2-indexed vector store built in-memory from chunk embeddings.
- **Question answering** — User queries embedded and searched against FAISS index; top-k chunks passed to Mistral API.
- **Mistral API integration** — `mistral-tiny` model used for generation via REST API.
- **Anti-hallucination prompt** — Strict instruction: answer only from context, or return `ANSWER_NOT_IN_DOCUMENTS`.
- **3-outcome response model** — PASS (grounded answer), Insufficient Evidence (SME escalation), No Evidence (SME escalation).
- **Citation formatting** — Harvard-style source citations appended to every PASS response via `citation_formatter.py`.
- **Export** — Answers exportable as TXT, JSON, and PDF (via ReportLab).
- **Confidence score** — FAISS-distance-derived score displayed per answer.
- **Streamlit UI** — Sidebar document upload, main chat interface, wide layout.

#### Supporting Modules
- **`citation_formatter.py`** — Formats source filenames into Harvard `(filename, n.d.)` citation strings.

#### Project Files
- `requirements.txt` — Initial dependency list.
- `README.md` — Basic usage documentation.
- `.gitignore` — Standard Python exclusions.
- `LICENSE` — MIT.

---

*Genie is built for enterprise document intelligence. Each release is defined by what it prevents, not just what it adds.*
