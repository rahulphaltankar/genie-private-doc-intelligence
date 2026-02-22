# genie/per_output_validator.py
from grounding import compute_grounding_score   # you already have this
# Removed citation validator import for now, as it's not strictly necessary for MCQ validation here, and might raise undefined errors if not present.
import logging

_logger = logging.getLogger(__name__)

# thresholds - tune them
STRICT_GROUNDING_THRESHOLD = 0.35   # for extractive MCQs, require stronger grounding
SYNTH_GROUNDING_THRESHOLD = 0.25    # for comprehension mode (if used elsewhere)

def validate_mcq(mcq: dict, chunks_store):
    """
    mcq: {question, options, answer, source:{doc_id, page, chunk_id}}
    chunks_store: a mapping chunk_id -> chunk_obj (with .text)
    returns: (bool, reason, grounding_score)
    """
    src = mcq.get("source", {})
    chunk_id = src.get("chunk_id")
    if not chunk_id:
        return False, "no source chunk_id", 0.0

    chunk = chunks_store.get(chunk_id)
    if not chunk:
        return False, "source chunk not in store", 0.0

    answer_text = mcq["answer"].strip()
    # direct literal check (conservative)
    if answer_text in chunk.text:
        # compute grounding score for audit
        score = compute_grounding_score(answer_text, [chunk.text])
        if score >= STRICT_GROUNDING_THRESHOLD:
            return True, "ok", score
        else:
            return False, f"low grounding {score:.3f}", score
    else:
        # no literal match -> fail
        score = compute_grounding_score(answer_text, [chunk.text])
        return False, "answer not literal in chunk", score
