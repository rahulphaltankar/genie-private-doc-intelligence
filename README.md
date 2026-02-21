# Genie AI — Premium Private Document Intelligence

Genie AI is a production-grade private document intelligence assistant that handles complex document tasks with strict grounding and premium aesthetics.

## Key Capabilities

- **Genie Wizard UX** — Intuitive sequential flow: Upload → Index → Chat
- **Universal Assistant** — Flexible task following (Summaries, Tables, Math, QA)
- **Production Retrieval** — Hybrid Search (FAISS + BM25) + Cross-Encoder Reranking
- **Rich Rendering** — Native support for LaTeX math and Markdown tables
- **Citation Enforcement** — Harvard-style references with page numbers
- **Grounding & Security** — Mandatory score validation and Audit logging

## Architecture Overview

```
User Instruction
→ Intent Detection (Universal vs. Specialized)
→ Hybrid Retrieval (FAISS + BM25)
→ Cross-Encoder Reranking
→ Instruction Execution (Mistral-7B Class)
→ Grounding Validation
→ Gatekeeper Decision
→ Rich UI Rendering (Table/Math)
→ Audit Logging
```

## Decision States

| Decision | Meaning |
|---|---|
| ✅ PASS | Grounded answer with valid inline citations |
| ⚠️ SYNTHESIS | Document-based synthesis without direct citation |
| 🚫 BLOCK | Insufficient evidence — SME escalation required |

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_api_key_here
```

## Run Locally

```bash
streamlit run app.py
```

## Project Structure

```
genie/
├── app.py                  # Main Streamlit application (Redesigned)
├── ingestion_pipeline.py    # Multi-format ingestion with page preservation
├── chunker.py               # Semantic page-aware chunking
├── hybrid_retriever.py      # Vector + BM25 merging logic
├── bm25_index.py            # Keyword search index
├── reranker.py              # Cross-Encoder scoring
├── mode_router.py           # Task intent detection
├── quiz_generator.py        # Specialized MCQ generation
├── grounding.py             # Grounding score computation
├── citation_validator.py    # Citation detection
├── gatekeeper.py            # Dual-enforcement decision engine
├── trace_logger.py          # Persistent audit logging
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── .gitignore
```

## Security Model

- **Citation-Enforced Answers** — No answer passes without source attribution
- **Grounding Gate** — Answer must be semantically derived from documents
- **No Silent Hallucination** — Refusals are explicit with reason
- **Audit Trace Logging** — Every decision persisted to `genie_trace_log.jsonl`
- **Zero-Trust UI** — Provenance and results are always distinct

## Supported File Formats

PDF, DOCX, TXT.

## License

MIT License
