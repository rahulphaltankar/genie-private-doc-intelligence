# Changelog

All notable changes to Genie are documented here.

---

## [v3.1.1] — 2026-02-22 — Final Release (Hackathon 3)

### Added
- **ISTQB AI QA Grounding Suite**
  - Added `istqb_test_execution.py` to systematically validate programmatic retrieval bounds, metric performance, threshold logic, and gatekeeper rule fidelity against baseline document sets.
  - Added `istqb_functional_test.py` to bypass GUI constraints and blast the LLM Mistral pipeline with 6 distinct classes of interrogative, formatting, and prompt-override trap test vectors. 

### Improved
- **Micro-UI Premium Polish**
  - Obliterated the artifact Streamlit border-radii bleeding through the chat input container via brute-force `!important` pseudo-element clearing.
  - Maximized contrast for the invisible chat send submit marker, elevating UX.
  - Flawless vertical and horizontal flex-box anchoring for the `genie · active` tag and the floating Document Knowledge pills.

---

## [v3.1.0] — 2026-02-22 — Hackathon 3 Final: The Comprehension Engine

### Added
- **Automated Quiz Generation** — `quiz_generator.py` orchestrates the creation of strict, document-bound MCQs from extracted facts.
- **Smart Distractor Generation** — Tricky, contextual distractors that use same-type facts from the document, preventing ungrounded hallucinations.
- **Structured Data Extraction** — Added `structured_extractor.py` to prompt the LLM to return strictly parsed facts.
- **Per-Output Validation** — Added `per_output_validator.py` to ensure every single generated question and distractor is grounded in the retrieved text.
- **Multi-chunk Synthesis Pipeline** — Connected the Streamlit UI to handle heavy document testing and render the AI-generated quiz seamlessly.

---

## [v3.0.0] — 2026-02-21 — Hackathon 3: The Universal Expansion

### Added
- **Universal Assistant Mode** — Genie now follows flexible instructions (Summarize, Table, Solve Math) using powerful system prompting.
- **Sequential Genie Wizard** — Redesigned UI with a 3-stage flow (Upload → Index → Chat).
- **Hybrid Search Engine** — Merged Vector (FAISS) and Keyword (BM25) search for robust retrieval.
- **Cross-Encoder Reranking** — Precision candidate refinement using MS-Marco MiniLM.
- **Rich Rendering** — Native LaTeX math support and Markdown table rendering in the chat view.
- **Page-Aware Ingestion** — Citations now include specific page numbers for enhanced auditability.

### Improved
- **Premium Aesthetics** — Vibrant Purple/Magenta theme with radial gradients and improved desktop sizing.
- **Intent Router** — Transitioned from rigid mode-switching to flexible task augmentation.

### Fixed
- **NLTK `punkt_tab` Error** — Resolved runtime dependency issue for sentence splitting.
- **Quiz `TypeError`** — Added robust sanitization for MCQ options parsing.

---

## [v2.1.0] — 2026-02-21 — Strict Enforcement Update

### Added
- **Unified Enterprise Decision Model** — Refactored `run_gatekeeper` to prioritize citation presence as a mandatory gate.
- **Hierarchical Outcomes** — Implemented tiered outcome selection:
  - If no citations → `BLOCK`
  - Score ≥ 0.40 → `PASS`
  - Score ≥ 0.25 → `SYNTHESIS`
  - Score < 0.25 → `BLOCK`
- **Exhaustive Fallback** — Updated `app.py` to ensure all non-PASS factual results trigger a comprehension fallback attempt before final blocking.

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
