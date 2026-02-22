import re

def detect_mode(query: str) -> str:
    query_lower = query.lower()
    
    # Keep specialized quiz generation as it uses a multi-step fact-extraction process
    if any(word in query_lower for word in ["quiz", "mcq", "multiple choice", "test me"]):
        return "quiz"
        
    comprehension_keywords = [
        "explain",
        "summarize",
        "describe",
        "overview",
        "architecture",
        "how does",
        "what is the purpose",
        "concept"
    ]
    
    for word in comprehension_keywords:
        if word in query_lower:
            return "comprehension"
            
    return "factual"
