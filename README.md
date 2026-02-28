# genie

> **Simplistic, premium document intelligence.**

An enterprise-grade document assistant that sacrifices hallucination for safety. Genie forces strict grounding mathematics and dual-lock citation verification on every inference, ensuring that if it doesn't exist in the document, it won't exist in the chat. 

Try it live: [https://genie-private-doc-intelligence-kns4yqtsqq6hshs7gq59m7.streamlit.app/](https://genie-private-doc-intelligence-kns4yqtsqq6hshs7gq59m7.streamlit.app/)

---

### The Problem

Structural bottlenecks exist when critical knowledge lives exclusively in unread PDFs or the heads of subject matter experts. Generic AI agents are dangerous in these constrained domains because they eagerly synthesize unverified facts. Genie solves this by turning dormant documentation into a trusted, **citation-enforced** intelligence layer. If an answer cannot be explicitly mapped to a retrieved paragraph, the Gatekeeper destroys the response before the user ever sees it.

---

### Architecture 

Genie employs a deterministic Retrieval-Augmented Generation (RAG) pipeline backed by `Mistral-7B-Instruct`, operating strictly as a factual proxy. 

**The Pipeline:**
1. **Semantic Ingestion:** Documents are parsed (`pdfplumber`) and mapped to an embedding lattice (`all-MiniLM-L6-v2`) while preserving strict multi-page provenance.
2. **Hybrid Search:** Queries execute simultaneously across dense `FAISS` vectors (semantic) and sparse `BM25` indices (keyword), maximizing edge-case recall.
3. **Cross-Encoder Reranking:** A `ms-marco` cross-encoder computationally rescores the retrieved chunks against the exact query intent to prioritize ground truth.
4. **Dual-Lock Gatekeeping (The Safety Layer):** 
    - *Gate 1 (Math):* Cosine similarity of the generation vs. the source chunks must exceed the defined grounding threshold (e.g., `0.40 FACTUAL`).
    - *Gate 2 (Regex):* The generation MUST contain a strict Harvard-style `(Filename, Page N)` citation anchor. 
    - **Result:** If either gate fails, the system executes an uncompromising `BLOCK`.

---

### Supported Workloads

Genie dynamically routes context based on the query's structural intent.

- **Factual Analytics:** Direct question answering with multi-source bounding. 
- **Tabular Extractions:** Forcing the LLM to output rigid Markdown and LaTeX tables.
- **Automated QA Engineering:** `quiz_generator.py` programmatically structures chunks into verified Multiple Choice quizzes using standalone NLI evaluation logic.

---

### Local Initialization

The project utilizes a standard Python virtual environment.

```bash
git clone https://github.com/rahulphaltankar/genie-private-doc-intelligence.git
cd genie-private-doc-intelligence
pip install -r requirements.txt
```

**Environment Config (`.env`):**
```bash
GEMINI_API_KEY=your_key_here
```

**Execute Node:**
```bash
streamlit run app.py
```

---

### Testing Instrumentation

The codebase contains a built-in programmatic ISTQB Quality Assurance harness. The system can be mathematically evaluated without requiring UI interactions. 

- `istqb_test_execution.py`: Validates component metrics (FAISS bounds, BM25 indices, `pdfplumber` recovery, Gatekeeper semantic traps).
- `istqb_functional_test.py`: Triggers deterministic adversarial prompt injections and formatting overloads against the LLM architecture.

---

### 📑 Hackathon 3 Documentation
For a deep-dive into the technical and functional evolution of Genie, see:
- [Techno-Functional Summary](docs/Techno_Functional_Summary.md)
- [Retrieval Optimization Walkthrough](docs/retrieval_walkthrough.md)

---

*genie is licensed under the MIT Protocol.*
