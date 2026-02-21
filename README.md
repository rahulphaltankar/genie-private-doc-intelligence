# Genie — Citation-Enforced Private Document Intelligence Assistant

Genie is an enterprise-grade private document intelligence system that provides grounded, citation-enforced answers over proprietary document collections.

## Core Capabilities

- Retrieval-Augmented Generation (RAG)
- Citation enforcement with Harvard-style references
- Grounding score validation
- Gatekeeper decision engine (PASS / SYNTHESIS / BLOCK)
- Hallucination prevention by design
- Audit logging for traceability
- SME escalation when evidence insufficient

## Architecture Overview

```
User Query
→ FAISS Retrieval
→ LLM Generation (Mistral)
→ Grounding Score Computation
→ Citation Validation
→ Gatekeeper Enforcement
→ Decision Output
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
├── app.py                  # Main Streamlit application
├── grounding.py            # Cosine similarity grounding score computation
├── citation_validator.py   # Harvard citation detection and validation
├── citation_formatter.py   # Harvard citation formatting
├── gatekeeper.py           # Dual-enforcement decision engine
├── trace_logger.py         # Persistent JSONL audit logging
├── requirements.txt
├── README.md
└── .gitignore
```

## Security Model

- Citation-enforced answers — no answer passes without source attribution
- Grounding score gate — answer must be semantically derived from documents
- No silent hallucination — every refusal is explicit with reason
- Full audit trace logging — every decision persisted to `genie_trace_log.jsonl`
- Explicit SME escalation when evidence is insufficient

## Supported File Formats

PDF, DOCX, TXT (up to 3 files, 500MB per file)

## License

MIT License
