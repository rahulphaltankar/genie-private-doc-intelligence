import re
from typing import List, Dict, Any
from metadata_schema import ChunkMeta

def extract_definitions(chunks: List[ChunkMeta]) -> List[Dict[str, Any]]:
    """
    Extracts simple definitions using regex heuristics.
    """
    definitions = []
    # Pattern: "Term is/are defined as..." or "Term refers to..."
    pattern = r"\b([A-Z][\w\s\-]+)\s+(?:is|are)\s+(?:defined as|referred to as|the)\s+([^.]+)\."
    
    for chunk in chunks:
        matches = re.finditer(pattern, chunk.text)
        for match in matches:
            definitions.append({
                "term": match.group(1).strip(),
                "definition": match.group(2).strip(),
                "source": {"filename": chunk.filename, "page": chunk.page, "chunk_id": chunk.chunk_id}
            })
    return definitions

def extract_equations(chunks: List[ChunkMeta]) -> List[Dict[str, Any]]:
    """
    Extracts equations using common symbols/syntax.
    """
    equations = []
    # Look for common math symbols or LaTeX-like syntax
    # Pattern: Looking for something with = and multiple math operators
    pattern = r"([^.\n]*?[=<>][^.\n]*?[\+\-\*/\^][^.\n]*)"
    
    for chunk in chunks:
        matches = re.findall(pattern, chunk.text)
        for match in matches:
            if len(match) > 10: # Simple filter for noise
                equations.append({
                    "equation": match.strip(),
                    "source": {"filename": chunk.filename, "page": chunk.page, "chunk_id": chunk.chunk_id}
                })
    return equations

def extract_factual_sentences(chunks: List[ChunkMeta]) -> List[Dict[str, Any]]:
    """
    Extracts sentences containing numerical facts or specific named entities.
    """
    facts = []
    # Pattern: Sentence with a number and a noun
    num_pattern = r"([^.]*?\d+(?:\.\d+)?%?\s+\w+[^.]*\.)"
    
    for chunk in chunks:
        matches = re.findall(num_pattern, chunk.text)
        for match in matches:
            facts.append({
                "text": match.strip(),
                "source": {"filename": chunk.filename, "page": chunk.page, "chunk_id": chunk.chunk_id}
            })
    return facts
