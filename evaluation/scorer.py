import re

def score_retrieval(expected_chunk_identifiers, retrieved_chunk_identifiers):
    """
    Weighted Retrieval Score (Max 30)
    - Expected source in top 3 = 30
    - In top 5 = 20
    - Present anywhere = 10
    - Else = 0
    """
    if not expected_chunk_identifiers or not retrieved_chunk_identifiers:
        return 0
        
    for expected in expected_chunk_identifiers:
        if expected in retrieved_chunk_identifiers[:3]: return 30
        if expected in retrieved_chunk_identifiers[:5]: return 20
        if expected in retrieved_chunk_identifiers: return 10
    return 0

def score_grounding(hallucinated_claim, minor_unsupported, is_correct_block):
    """
    Weighted Grounding Score (Max 35)
    - Hallucinated claim = 0
    - Minor unsupported = 25
    - Fully supported = 35
    - Correct BLOCK = 35
    """
    if is_correct_block:
        return 35
    if hallucinated_claim:
        return 0
    if minor_unsupported:
        return 25
    return 35

def score_citation(llm_answer, required_citation_count, actual_citation_count):
    """
    Weighted Citation Score (Max 15)
    - All claims cited = 15
    - Partial = 10
    - Missing = 0
    """
    if required_citation_count == 0:
        return 15 # N/A if not factual
        
    if actual_citation_count == 0:
        return 0
    if actual_citation_count >= required_citation_count:
        return 15
    return 10

def score_structure(llm_answer, expected_type):
    """
    Weighted Structure Score (Max 10)
    - Matches expected output type = 10
    - Partial = 5
    - Wrong = 0
    """
    if expected_type == "BLOCK" or expected_type == "TEXT":
        return 10
        
    normalized = llm_answer.lower()
    
    if expected_type == "TABLE":
        if "|" in normalized and "-" in normalized: return 10
        return 0
    if expected_type == "JSON":
        if "{" in normalized and "}" in normalized and '"' in normalized: return 10
        return 0
    if expected_type == "QUIZ":
        if "a)" in normalized or "1." in normalized: return 10
        return 0
    if expected_type == "CHECKLIST" or expected_type == "MARKDOWN":
        if "- " in normalized or "* " in normalized: return 10
        return 5
        
    return 0

def score_decision(actual_decision, expected_behavior):
    """
    Decision Correctness (Max 10)
    - Correct BLOCK/ANSWER decision = 10
    - Wrong decision = 0
    """
    if expected_behavior == "BLOCK" and actual_decision == "BLOCK":
        return 10
    if expected_behavior in ["ANSWER", "STRUCTURED_ANSWER"] and actual_decision in ["PASS", "SYNTHESIS"]:
        return 10
    return 0

def calculate_total_score(retrieval_sc, grounding_sc, citation_sc, structure_sc, decision_sc):
    return retrieval_sc + grounding_sc + citation_sc + structure_sc + decision_sc
