"""
gatekeeper.py — Authoritative dual-enforcement layer for Genie.

Enforces BOTH conditions before allowing an answer through:
    1. grounding_score >= threshold  (semantic similarity to retrieved chunks)
    2. citation_present              (inline source reference in the LLM response)

Grounding similarity alone is not sufficient — a paraphrased answer can score
high similarity but contain no verifiable source attribution. Citation presence
is the second independent gate that closes this gap.

Decision table:
    Factual mode:
        citation_present AND grounding_score >= FACTUAL_PASS_THRESHOLD  →  PASS
        else                                                             →  BLOCK

    Synthesis mode:
        citation_present AND grounding_score >= SYNTHESIS_PASS_THRESHOLD  →  SYNTHESIS
        else                                                               →  BLOCK
"""

from typing import Literal
from citation_validator import has_valid_citations

GatekeeperDecision = Literal["PASS", "SYNTHESIS", "BLOCK"]

# Grounding score thresholds (cosine similarity, 0.0–1.0)
FACTUAL_PASS_THRESHOLD = 0.45
SYNTHESIS_PASS_THRESHOLD = 0.30


def run_gatekeeper(
    answer: str,
    retrieved_chunks: list,
    grounding_score: float,
    mode: Literal["factual", "synthesis"] = "factual"
) -> tuple:
    """
    Unified enterprise gatekeeper logic.
    
    Decision Model:
    1. If no citations present → BLOCK
    2. Else if grounding_score >= FACTUAL_PASS_THRESHOLD → PASS
    3. Else if grounding_score >= SYNTHESIS_PASS_THRESHOLD → SYNTHESIS
    4. Else → BLOCK
    """

    # ── Gate 0: Sentinel check ───────────────────────────────────────────
    if not answer or "ANSWER_NOT_IN_DOCUMENTS" in answer:
        return ("BLOCK", "No grounded answer found in uploaded documents.")

    # ── Gate 1: Citation Presence (Mandatory) ────────────────────────────
    if not has_valid_citations(answer):
        return ("BLOCK", "No direct citations found in answer. All responses must be grounded.")

    # ── Gate 2: Tiered Logic ──────────────────────────────────────────────
    if mode == "comprehension":
        if grounding_score >= SYNTHESIS_PASS_THRESHOLD:
            return ("SYNTHESIS", f"SYNTHESIS — Synthesised answer (score {grounding_score:.3f})")
        else:
            return ("BLOCK", f"Low grounding score ({grounding_score:.3f} < {SYNTHESIS_PASS_THRESHOLD}). Answer not sufficiently derived from documents.")
    else:
        if grounding_score >= FACTUAL_PASS_THRESHOLD:
            return ("PASS", f"PASS — Grounded answer (score {grounding_score:.3f})")
        else:
            return ("BLOCK", f"Low grounding score ({grounding_score:.3f} < {FACTUAL_PASS_THRESHOLD}). Answer not sufficiently derived from documents.")
