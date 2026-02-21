import re

def detect_intent(query: str) -> str:
    """
    Detects if a specialized pipeline (like Quiz) is needed.
    Otherwise, defaults to 'universal' for flexible instruction following.
    """
    query_lower = query.lower()
    
    # Keep specialized quiz generation as it uses a multi-step fact-extraction process
    if any(word in query_lower for word in ["quiz", "mcq", "multiple choice", "test me"]):
        return "quiz"
        
    return "universal"
