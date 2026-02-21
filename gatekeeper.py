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
FACTUAL_PASS_THRESHOLD = 0.40
SYNTHESIS_PASS_THRESHOLD = 0.25


def run_gatekeeper(
    answer: str,
    retrieved_chunks: list,
    grounding_score: float,
    mode: Literal["factual", "synthesis"] = "factual"
) -> tuple:
    """
    Dual-enforcement decision engine.

    Args:
        answer:           The LLM-generated answer string.
        retrieved_chunks: List of document text chunks used as context.
        grounding_score:  Cosine similarity between answer and chunks (0.0–1.0).
        mode:             "factual" (strict path) or "synthesis" (comprehension path).

    Returns:
        (decision, reason) where decision is PASS | SYNTHESIS | BLOCK
    """

    # ── Gate 0: Hard block on empty or sentinel responses ─────────────────
    if not answer or answer.strip() == "":
        return ("BLOCK", "Empty answer returned from language model.")

    if "ANSWER_NOT_IN_DOCUMENTS" in answer:
        return ("BLOCK", "Language model indicated answer is not in documents.")

    # ── Gate 1: Citation presence ──────────────────────────────────────────
    citation_present = has_valid_citations(answer)

    # ── Gate 2: Grounding score threshold ─────────────────────────────────
    if mode == "factual":
        grounding_ok = grounding_score >= FACTUAL_PASS_THRESHOLD

        if citation_present and grounding_ok:
            return ("PASS", f"PASS — citation present, grounding score {grounding_score:.3f}")

        # Diagnose specifically which gate failed for the BLOCK reason
        if not citation_present and not grounding_ok:
            return (
                "BLOCK",
                f"No inline citation found AND low grounding score ({grounding_score:.3f} < {FACTUAL_PASS_THRESHOLD}). "
                "Answer could not be verified against uploaded documents."
            )
        if not citation_present:
            return (
                "BLOCK",
                f"No inline citation found in answer (grounding score {grounding_score:.3f} was sufficient). "
                "Source provenance cannot be confirmed."
            )
        # grounding_ok is False
        return (
            "BLOCK",
            f"Low grounding score ({grounding_score:.3f} < {FACTUAL_PASS_THRESHOLD}). "
            "Answer not sufficiently derived from uploaded documents."
        )

    elif mode == "synthesis":
        grounding_ok = grounding_score >= SYNTHESIS_PASS_THRESHOLD

        if citation_present and grounding_ok:
            return ("SYNTHESIS", f"SYNTHESIS — citation present, grounding score {grounding_score:.3f}")

        if not citation_present and not grounding_ok:
            return (
                "BLOCK",
                f"No inline citation found AND low grounding score ({grounding_score:.3f} < {SYNTHESIS_PASS_THRESHOLD}). "
                "Answer appears to be from general knowledge only."
            )
        if not citation_present:
            return (
                "BLOCK",
                f"No inline citation found in synthesised answer (grounding score {grounding_score:.3f} was sufficient). "
                "Source provenance cannot be confirmed."
            )
        return (
            "BLOCK",
            f"Low grounding score ({grounding_score:.3f} < {SYNTHESIS_PASS_THRESHOLD}). "
            "Synthesised answer not sufficiently derived from uploaded documents."
        )

    # ── Fallback — unknown mode, block for safety ──────────────────────────
    return ("BLOCK", f"Unknown gatekeeper mode: {mode}")
