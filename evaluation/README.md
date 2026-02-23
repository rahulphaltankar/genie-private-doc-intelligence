# Genie Golden Evaluation Suite

This is the automated User Acceptance Testing (UAT) and regression harness for the Genie Document Intelligence platform.

## What it does
It headlessly executes 120 dynamic prompts (spread across compliance tasks, formatting challenges, and adversarial injections) against the core RAG architecture and scores the output based on a deterministic 100-point rubric:
* **Retrieval (30%):** Were the right chunks pulled?
* **Grounding (35%):** Is the answer mathematically derived from the document?
* **Citation (15%):** Are Harvard-style references properly formatted?
* **Structure (10%):** Did the LLM output the requested JSON/Markdown?
* **Decision (10%):** Did the Gatekeeper correctly PASS or BLOCK the request?

## How to Run the Suite

1. **Start the local backend** (Optional for test execution since it imports directly, but necessary if testing the frontend UI):
   ```bash
   streamlit run app.py
   ```
2. **Execute the Evaluation Harness:**
   This will take roughly 5-10 minutes depending on the Mistral API Latency.
   ```bash
   python evaluation/run_tests.py
   ```
3. **View the Report:**
   Once finished, open `evaluation/reports/latest_report.json` to review the precise scores.

## How to Interpret the CI Gate limits
Run the Continuous Integration binary:
```bash
python evaluation/ci_gate.py
```
If your codebase causes the system to drop below an `80` average score, hallucinate more than `2%` of the time, or fail to correctly `BLOCK` adversarial prompts `95%` of the time, the gate will explicitly fail and prevent production deployment.
