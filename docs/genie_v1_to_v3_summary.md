# Genie: Techno-Functional-Architectural-Contextual Summary
## Version 1.0 → 3.2.1 | Hackathons 1 through 3

> **Scope**: This document covers every version of Genie from its prototype inception through the final public release milestone at v3.2.1.  
> **Status**: v3 is publicly sanctioned; v4+ is commercially private (ref. `docs/SIGNOFF.md`).

---

## 1. Project Origin & Contextual Intent

Genie was conceived as a **private document intelligence assistant** — a tool that lets knowledge workers interrogate proprietary PDF/DOCX/TXT documents and receive answers that are **mathematically grounded**, **source-attributed**, and **audit-ready**, with zero tolerance for hallucination.

The product was built across three time-boxed hackathons, each representing a distinct evolutionary leap:

| Phase | Hackathon | Versions | Thematic Focus |
|---|---|---|---|
| **Foundation** | Hackathon 1 | v1.0.0 | File ingestion, FAISS vector search, basic Q&A via Mistral |
| **Trust Layer** | Hackathon 2 | v2.0.0 → v2.1.0 | Dual-gate enforcement, hallucination prevention by architecture |
| **Comprehension Engine** | Hackathon 3 | v3.0.0 → v3.2.1 | Hybrid retrieval, cross-encoder reranking, quiz generation pipeline |

The primary LLM throughout all versions is **Mistral (`mistral-small-latest`)**, accessed via the Mistral AI API. The UI is built on **Streamlit**. The embedding layer uses **`all-MiniLM-L6-v2`** (sentence-transformers) for semantic encoding.

---

## 2. Hackathon 1 — The Prototype Foundation (v1.0.0)

### 2.1 Functional Intent

Build a minimum viable document Q&A system: upload a document, ask a question, receive an answer drawn from that document.

### 2.2 Core Architecture

The v1 architecture was a single-file monolith (`app.py`) wrapping four basic operations:

```
User Query
    │
    ▼
FAISS Vector Index  ←── sentence-transformers all-MiniLM-L6-v2
    │
    ▼
Top-K Chunk Retrieval  (pure dense / semantic search, flat L2 index)
    │
    ▼
Mistral API  (context-stuffed prompt)
    │
    ▼
Raw Answer → UI
```

### 2.3 Ingestion & Chunking (v1 baseline)

- **Supported formats**: PDF, DOCX, TXT
- **PDF parsing**: `pypdf` `PdfReader` — per-page text extraction
- **DOCX parsing**: `python-docx` — full document as single text stream
- **Chunking**: Sentence-aware via NLTK `sent_tokenize`, with a 1,500-character threshold. Each chunk carries a **2-sentence overlap window** for context continuity across boundaries
- **Metadata schema** (`ChunkMeta` dataclass): `chunk_id` (UUID), `doc_id`, `filename`, `page`, `title`, `author`, `text`, `tokens`, `created_at`

### 2.4 Vector Search

- **Index type**: FAISS `IndexFlatL2` — exact L2 distance, no compression
- **Embedding model**: `all-MiniLM-L6-v2` (384-dimensional dense embeddings)
- **Index is non-persistent** — rebuilt on each session from uploaded files

### 2.5 Prompt Design

System prompt instructed Mistral to:
1. Use ONLY the provided context
2. Return `"ANSWER_NOT_IN_DOCUMENTS"` if the context is insufficient
3. Include inline Harvard-style citations `(Filename, Page X)`

### 2.6 Intent Routing (v1)

Intent routing in v1 was primitive — keyword-based matching for quiz/comprehension triggers. FAISS distance-threshold routing was introduced in v2.

### 2.7 Limitations of v1

| Gap | Impact |
|---|---|
| No grounding score | No way to quantify answer fidelity |
| No citation validation | LLM could produce answers without citing sources |
| No audit log | Zero auditability |
| No hallucination enforcement | Trust entirely delegated to LLM instruction |
| Pure dense search | Weak on exact terms (article numbers, legal codes) |
| Single-stage retrieval | Top-K chunks passed directly to LLM |

---

## 3. Hackathon 2 — The Trust Enforcement Layer (v2.0.0 → v2.1.0)

### 3.1 Thematic Shift

> *"Hallucination prevention is now enforced by architecture, not by prompt instruction."*

The core insight of Hackathon 2 was that **LLM prompts are inherently persuadable** — a sufficiently crafted user query can bypass instruction-level guardrails. The solution: enforce grounding at the infrastructure layer, downstream of the LLM, as a deterministic post-processing gate.

### 3.2 New Modules Introduced

| Module | Role |
|---|---|
| `gatekeeper.py` | Dual-enforcement decision engine |
| `grounding.py` | Cosine similarity grounding scorer |
| `citation_validator.py` | Harvard-style citation regex validator |
| `trace_logger.py` | Persistent JSONL audit logger |
| `mode_router.py` | FAISS distance-threshold intent router |

---

### 3.3 Grounding Score Engine — `grounding.py`

The grounding score measures **semantic similarity between the LLM-generated answer and the retrieved source chunks**:

```python
answer_embedding    = model.encode(answer)        # all-MiniLM-L6-v2
chunk_embeddings    = model.encode(retrieved_chunks)
grounding_score     = max(cosine_similarity(answer_embedding, chunk_embeddings))
```

- Returns a float in `[0.0, 1.0]`  
- A high score means the answer is semantically close to at least one retrieved chunk  
- This is **independent of the LLM's instruction compliance** — a paraphrased hallucination can still score high, which is why a second gate (citation) was added

---

### 3.4 Citation Validator — `citation_validator.py`

A regex-based validator requiring the presence of at least one Harvard-style inline citation in every LLM response.

Three supported citation formats (evolved across versions):

| Pattern | Format | Example |
|---|---|---|
| Harvard Full | `(Author, Year, p. N)` | `(Bank of England, 2024, p. 4)` |
| Harvard n.d. | `(Filename, n.d.)` | `(policy_document.pdf, n.d.)` |
| Page-Aware (v3+) | `(Filename, Page N)` | `(regulation.pdf, Page 12)` |

The v3.2.1 patch extended patterns to support markdown styling (`*italics*`, `**bold**`) inside citation brackets, reducing false-block rate by ~15%.

---

### 3.5 The Dual-Gate Gatekeeper — `gatekeeper.py`

The `run_gatekeeper()` function is the authoritative enforcement boundary. It evaluates every LLM response against two **independent and mandatory** conditions before any answer is allowed through:

```
Gatekeeper Decision Model:

Gate 0: Sentinel Check
  └── if answer empty OR contains "ANSWER_NOT_IN_DOCUMENTS" → BLOCK

Gate 1: Citation Presence (Mandatory)
  └── if no valid Harvard citation found → BLOCK

Gate 2: Tiered Grounding Score
  ├── mode == "factual":
  │     grounding_score ≥ 0.55 → PASS
  │     else → BLOCK
  └── mode == "comprehension":
        grounding_score ≥ 0.40 → SYNTHESIS
        else → BLOCK
```

**Decision States** (always surfaced in the UI):
- ✅ **PASS** — Fully grounded + citation-verified. Score ≥ 0.55
- ⚠️ **SYNTHESIS** — Grounded synthesis across multi-chunk context. Score ≥ 0.40
- 🚫 **BLOCK** — Answer cannot be verified. SME escalation triggered

The thresholds (`FACTUAL_PASS_THRESHOLD = 0.55`, `SYNTHESIS_PASS_THRESHOLD = 0.40`) were tuned through the Hackathon 3 threshold sweep.

---

### 3.6 Audit Logger — `trace_logger.py`

Every query — successful or blocked — is appended to `genie_trace_log.jsonl`:

```json
{
  "timestamp": "2026-02-21T12:00:00.000Z",
  "query": "What is Article 6?",
  "answer": "...",
  "grounding_score": 0.72,
  "decision": "PASS",
  "sources": ["regulation.pdf"]
}
```

This makes Genie auditable at the infrastructure level — a prerequisite for enterprise deployment where decisions must be explainable.

---

### 3.7 Intent Router (v2 Upgrade) — `mode_router.py`

v2 replaced brittle keyword matching with **FAISS distance-threshold routing** and later with a keyword-based but semantically organized router:

```
Query analysis:
  "quiz" / "mcq" / "test me"                        → quiz mode
  "explain" / "summarize" / "overview" / "how does"  → comprehension mode
  default                                             → factual mode
```

- **Factual mode**: Tight single-source bounding. FACTUAL_PASS_THRESHOLD = 0.55
- **Comprehension mode**: Multi-chunk synthesis permitted. SYNTHESIS_PASS_THRESHOLD = 0.40
- **Quiz mode**: Full multi-step fact-extraction pipeline (v3)

---

### 3.8 v2.1.0 — Strict Enforcement Update

v2.1.0 hardened the gatekeeper further:

- **Citations became a mandatory prerequisite**, not just a scoring signal — any answer without a citation is immediately blocked regardless of grounding score
- SYNTHESIS was restricted to comprehension mode only; factual mode had no SYNTHESIS path (PASS or BLOCK only)
- `app.py` was updated to trigger a **comprehension fallback** for all non-PASS factual results before issuing a final BLOCK

---

## 4. Hackathon 3 — The Comprehension Engine (v3.0.0 → v3.2.1)

### 4.1 Thematic Shift

Hackathon 3 redefined Genie from a Q&A assistant to a **multi-modal document intelligence platform**. Three fundamental upgrades were shipped:

1. **Retrieval quality** — Hybrid search (vector + keyword) with cross-encoder reranking
2. **Cognition depth** — Atomic fact extraction + automated MCQ generation with per-output grounding validation
3. **Production hardening** — Systematic recall/precision optimization harness, ISTQB test suite, UI polish

### 4.2 New Modules Introduced

| Module | Role |
|---|---|
| `bm25_index.py` | Sparse BM25 keyword index |
| `hybrid_retriever.py` | Weighted fusion of dense FAISS + sparse BM25 |
| `reranker.py` | Cross-encoder MS-Marco precision reranker |
| `structured_extractor.py` | Atomic fact extraction from chunks |
| `quiz_generator.py` | MCQ generation from extracted facts |
| `per_output_validator.py` | Per-MCQ grounding enforcer |
| `evaluation/` | Full evaluation harness (metrics, sweep, CI gate) |

---

### 4.3 Hybrid Retrieval Engine

#### Dense Search — FAISS (`IndexFlatL2`)

- Query embedded via `all-MiniLM-L6-v2` (384d)  
- L2 distance search fetches `top_k × 2` candidates for the reranking pool  
- L2 distances **inverted and min-max normalized** to [0, 1] (lower distance = higher relevance)

#### Sparse Search — BM25 (`bm25_index.py`)

- Uses `rank_bm25.BM25Okapi` library  
- Index built from tokenized chunk texts (lowercased, whitespace-split)  
- `score_all(query)` returns raw BM25 scores for every chunk  
- Scores min-max normalized to [0, 1]

#### Score Fusion — `hybrid_retriever.py`

```python
final_score[i] = alpha * vector_score_norm[i] + (1.0 - alpha) * bm25_score_norm[i]
```

- `alpha = 0.5` (50/50 dense/sparse balance) — the winning configuration from sweep
- Optional `metadata_boost`: structural header chunks (text < 100 chars or title-matching) receive a score bonus to anchor retrieval in document structure
- Top-K sorted by `final_score` descending

**Why hybrid?** Dense search excels at semantically-phrased questions; BM25 excels at exact-token requests (article numbers, legal codes, unique identifiers). The fusion improves both precision and recall simultaneously.

#### Cross-Encoder Reranking — `reranker.py`

After hybrid retrieval produces a pool of `top_k` candidates, a **cross-encoder** rescores them:

```
Cross-Encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
Input: (query, chunk_text) pairs
Output: relevance scores (higher = more relevant)
```

Unlike bi-encoders (which embed query and chunks separately), cross-encoders **process the query and chunk jointly**, allowing full attention across both — producing significantly more precise relevance scores at the cost of O(k) inference calls. The pool depth was set to **Top 10** (v3.2.0) after sweep analysis showed this maximized average score without excessive latency.

---

### 4.4 Three-Stage Wizard UI

v3 redesigned the Streamlit application from a monolithic page to a **sequential three-stage wizard**:

```
Stage 1: Upload  →  Stage 2: Indexing  →  Stage 3: Chat
```

- **Upload**: Multi-file drag-and-drop. Files are staged in `st.session_state`
- **Indexing**: On "Build Assistant Brain" click — FAISS index built, BM25 index built, `ChunkMeta` list stored in session. Both indices are built **concurrently within the same spinner** (sequential in implementation but presented as atomic)
- **Chat**: Full conversation history, per-message download buttons (.md and .pdf), floating "Document Knowledge" pills showing active source files, compact `genie · active` header

A **Reset** button clears the entire session state, allowing fresh uploads without page reload.

---

### 4.5 Rich Rendering

v3.0.0 added two UI-level rendering capabilities:
- **LaTeX math**: `$$ ... $$` blocks rendered natively by Streamlit/Markdown for scientific/mathematical documents
- **Markdown tables**: Structured data returned by Mistral in table format is rendered correctly in the chat view

The system prompt instructs Mistral to always use these formats where relevant.

---

### 4.6 Automated MCQ Pipeline

The quiz mode pipeline is a 5-stage chain:

```
Stage 1: Hybrid Retrieval  (top_k=15 for quiz mode, broader pool)
    │
Stage 2: Cross-Encoder Reranking  (top 15 → re-ordered by relevance)
    │
Stage 3: Atomic Fact Extraction  [structured_extractor.py]
    │  Sentences classified as: equation / definition / method / result / numeric / fact
    │  Deduplication on fact text
    │
Stage 4: MCQ Generation  [quiz_generator.py]
    │  Facts used as correct answers
    │  Distractors drawn from OTHER facts in the same document (same type, different fact)
    │  Prevents hallucinated or trivially-wrong distractors
    │
Stage 5: Per-Output Grounding Validation  [per_output_validator.py]
    │  For each MCQ: answer_text must appear LITERALLY in its source chunk
    │  Grounding score must be ≥ 0.35 (STRICT_GROUNDING_THRESHOLD)
    │  Failed MCQs logged to genie_trace_log.jsonl as "BLOCK_MCQ"
    │
Final: Return top N validated MCQs to UI
```

**Key design decision**: Distractors are **document-native**, sourced from the same document's other facts. This makes distractors plausible and topic-coherent without hallucinating made-up alternatives.

---

### 4.7 Ingestion Pipeline (v3 upgrade)

`ingestion_pipeline.py` / `chunker.py` evolved to produce **page-aware chunks**:

- PDF: Per-page extraction. Each chunk carries its exact `page` number
- DOCX: Full document treated as a single page (section-aware future work)
- Chunking: NLTK sentence tokenizer, 1,500-char threshold, 2-sentence overlap
- Each `ChunkMeta` object carries: `chunk_id`, `doc_id`, `filename`, `page`, `text`, `tokens`, `created_at`

The `page` field enables **page-level citations** in format `(Filename, Page N)` — a v3 addition that replaced the vaguer `(Filename, n.d.)` pattern for freshly-ingested documents.

---

### 4.8 Evaluation Harness — `evaluation/`

v3.1.1 introduced a production-grade evaluation and CI framework:

| Component | Purpose |
|---|---|
| `run_tests.py` | Runs the full test suite against `test_suite.json` |
| `metrics.py` | Computes Recall@K, average score, hallucination rate, block accuracy |
| `scorer.py` | Per-question scoring logic |
| `generate_test_suite.py` | Utility to generate structured test cases |
| `retrieval_experiments.py` | Multi-dimensional sweep harness (k, alpha, expansion, boost, rerank depth) |
| `threshold_sweep.py` | Gatekeeper threshold optimization sweep |
| `report_generator.py` | Human-readable results reporter |
| `ci_gate.py` | CI/CD gate: fails if Recall@3 < 85% or hallucination rate > 5% |

**Sweep results** (`docs/retrieval_walkthrough.md`):

| Configuration | Recall@3 | Avg Score | Latency Δ |
|---|---|---|---|
| Baseline (k=5) | 100% | 88.3 | 0% |
| k=8 | 100% | 87.1 | +12% |
| k=10 | 100% | 87.1 | +0.2% |
| 0.7 Dense | 100% | 85.8 | +7.2% |
| 0.3 Dense | 100% | 87.5 | +0.2% |
| Context Expansion | 100% | 89.2 | +30.7% |
| **Rerank Top 10 ✓** | **100%** | **89.6** | **+25.1%** |
| Metadata Boost | 100% | 88.3 | +33.8% |
| Kitchen Sink | 100% | 87.1 | +71.8% |

Winner: **Rerank Top 10** — optimal score/latency tradeoff. Applied to production in v3.2.0.

---

### 4.9 ISTQB AI QA Testing Suite (v3.1.1)

Two dedicated test executors were added to provide systematic, structured QA coverage:

- **`istqb_functional_test.py`**: Bypasses the GUI and directly interrogates the Mistral pipeline with 6 classes of adversarial test vectors — interrogative, formatting, and prompt-override traps. Tests the system's resistance to injection and boundary pushing
- **`istqb_test_execution.py`**: Validates programmatic retrieval bounds, metric performance, threshold logic, and rule fidelity against known document baselines

---

### 4.10 UI Aesthetic — IDE Dark Theme

v3.1.1 finalized the premium visual identity — a VSCode-inspired IDE dark theme:

| Element | Value |
|---|---|
| Background | `#1e1e1e` (VSCode editor black) |
| Surface panels | `#252526` (VSCode sidebar) |
| Border | `#333333` (VSCode panel border) |
| Accent (links, focus) | `#007acc` (VSCode blue) |
| Text | `#d4d4d4` (VSCode muted white) |
| Code inline | `#ce9178` (VSCode string orange) |
| Table headers | `#9cdcfe` (VSCode variable blue) |
| Typography | Inter + JetBrains Mono (Google Fonts) |

Micro-UX fixes in v3.1.1: eliminated Streamlit border-radius bleeding from chat input, maximized send-button icon contrast, locked vertical/horizontal flexbox alignment for `genie · active` tag and knowledge pill row.

---

### 4.11 Citation Regex Patch (v3.2.1)

A production bug was identified where markdown-styled text inside citations (`*italic filename*`, `**bold**`) caused the citation regex to fail, triggering false BLOCK decisions. The patch updated `citation_validator.py` patterns to support `[\w\s\-\.\*]+` inside parentheses, reducing false-block rate by ~15%.

---

## 5. Full Module Map (v3.2.1)

```
genie/
├── app.py                    # Streamlit application entrypoint & UI orchestrator
├── ingestion_pipeline.py     # File upload handler (PDF/DOCX/TXT → ChunkMeta list)
├── chunker.py                # Page-aware, sentence-split, overlapping chunker
├── metadata_schema.py        # ChunkMeta dataclass definition
├── bm25_index.py             # BM25Okapi sparse keyword index wrapper
├── hybrid_retriever.py       # Weighted FAISS+BM25 fusion retriever (alpha, boost)
├── reranker.py               # Cross-encoder MS-Marco reranker (graceful fallback)
├── mode_router.py            # Intent classifier (factual / comprehension / quiz)
├── grounding.py              # Cosine similarity grounding scorer (all-MiniLM-L6-v2)
├── gatekeeper.py             # Dual-gate enforcement (sentinel → citation → score)
├── citation_validator.py     # Harvard-style citation regex extractor & validator
├── citation_formatter.py     # Harvard citation string formatter
├── trace_logger.py           # Persistent JSONL audit log writer
├── structured_extractor.py   # Atomic fact extractor (sentence classification)
├── quiz_generator.py         # MCQ generator from extracted facts + distractor logic
├── per_output_validator.py   # Per-MCQ grounding enforcer (literal + cosine gate)
├── evaluation.py             # Lightweight eval runner (legacy)
├── evaluation/               # Full evaluation harness
│   ├── run_tests.py          # Test executor
│   ├── metrics.py            # Recall@K, hallucination rate, block accuracy
│   ├── scorer.py             # Per-question scorer
│   ├── retrieval_experiments.py  # Multi-dimensional retrieval sweep
│   ├── threshold_sweep.py    # Gatekeeper threshold optimizer
│   ├── report_generator.py   # Human-readable report builder
│   ├── ci_gate.py            # CI/CD quality gate
│   └── test_suite.json       # Baseline AI Act evaluation cases
├── docs/
│   ├── SIGNOFF.md            # Founder signoff on v3 public / v4+ private policy
│   ├── Techno_Functional_Summary.md  # Hackathon 3 summary
│   ├── retrieval_walkthrough.md      # Sweep results & winning config
│   └── USER_SOP.md           # End-user operating procedure
├── genie_trace_log.jsonl     # Live audit log (append-only)
├── requirements.txt          # Python dependency manifest
└── tests/                    # Unit tests
    └── test_quiz_pipeline.py # Quiz pipeline test coverage
```

---

## 6. Data Flow — End-to-End Request Lifecycle (v3.2.x)

```
User Types Query
        │
        ▼
mode_router.detect_mode(query)
  → factual / comprehension / quiz
        │
        ▼
hybrid_retriever.hybrid_search(
  query, vector_store, bm25_index, chunks,
  top_k = 10 (factual/comprehension) | 15 (quiz),
  alpha = 0.5
)
  → candidate chunk indices
        │
        ▼
Reranker.rerank(query, candidate_chunks)
  → sorted [(ChunkMeta, CrossEncoder score)]
        │
   ┌────┴────┐
   │         │
[quiz]    [factual / comprehension]
   │         │
   ▼         ▼
structured_extractor    Mistral API
.extract_atomic_facts     (context = top 10 chunks as
  → fact list              "Source (filename, Page N): text")
   │         │
   ▼         ▼
quiz_generator          grounding.compute_grounding_score
.generate_mcqs_from_facts  → cosine_sim(answer, chunks)
   │         │
   ▼         ▼
per_output_validator    gatekeeper.run_gatekeeper
.validate_mcq             → PASS / SYNTHESIS / BLOCK
   │         │
   ▼         ▼
Final MCQs      Answer (or Block message)
        │
        ▼
trace_logger.log_trace(query, answer, score, decision, sources)
        │
        ▼
Streamlit Chat UI
(+ .md / .pdf download buttons)
```

---

## 7. Quality Metrics — Final State at v3.2.0

| Metric | Target | Achieved | Status |
|---|---|---|---|
| **Retrieval Recall@3** | ≥ 85% | **100.0%** | 🏆 Exceeded |
| **Average Score** | ≥ 80 | **89.6** | 🏆 Exceeded |
| **Hallucination Rate** | ≤ 5% | **0.0%** | 🏆 Zero-Hallucination |
| **Gatekeeper BLOCK Accuracy** | ≥ 95% | **100.0%** | 🏆 Exceeded |

Gatekeeper thresholds after optimization sweep:
- `FACTUAL_PASS_THRESHOLD = 0.55`
- `SYNTHESIS_PASS_THRESHOLD = 0.40`
- `MCQ STRICT_GROUNDING_THRESHOLD = 0.35`

---

## 8. Versioning Timeline

| Version | Date | Milestone |
|---|---|---|
| v1.0.0 | Hackathon 1 | FAISS prototype — ingest, embed, retrieve, answer |
| v2.0.0 | 2026-02-21 | Trust Enforcement Layer — gatekeeper, grounding, citations, audit log |
| v2.1.0 | 2026-02-21 | Strict enforcement — citation as mandatory gate, synthesis restricted |
| v3.0.0 | 2026-02-21 | Hybrid search, cross-encoder reranking, quiz mode, LaTeX/table rendering |
| v3.1.0 | 2026-02-22 | Comprehension Engine — quiz pipeline, MCQ validation, multi-chunk synthesis UI |
| v3.1.1 | 2026-02-22 | ISTQB QA suite, premium UI polish (Hackathon 3 final release) |
| v3.2.0 | 2026-02-23 | Retrieval optimization sweep — Recall@3 = 100%, Avg Score = 89.6 |
| v3.2.1 | 2026-02-28 | Citation regex robustness patch — markdown-styled citations supported |

---

## 9. Strategic Context & Forward Direction

v3 is formally designated as the **public reference artifact** — sanitized and open for community reference. The founder's signoff (`docs/SIGNOFF.md`, 2026-02-28) established:

- **v3 (Public)**: Full source exposure, all sensitive data removed
- **v4+ (Private)**: Commercially sensitive; source, models, and data remain proprietary
- **Pilot delivery**: Docker containers or offline bundles — no raw source exposure for enterprise clients

The v4 work (visible in `c:\Users\shrut\OneDrive\Desktop\RAG Corpus\V4`) extends Genie into enterprise-grade territory with features not covered by this document.

---

*Document prepared: 2026-03-07 | Covers: v1.0.0 → v3.2.1 | Canonical source: `docs/genie_v1_to_v3_summary.md`*
