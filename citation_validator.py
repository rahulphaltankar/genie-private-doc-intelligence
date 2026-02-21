"""
citation_validator.py — Validates citation presence in LLM answers.

Supports two citation formats:
  1. Harvard full:  (Author, Year, p. N)           e.g. (Bank of England, 2024, p. 4)
  2. Harvard n.d.:  (Filename, n.d.)               e.g. (policy_document.pdf, n.d.)
"""

import re

# Harvard full citation: (Author, 2024, p. 4)
HARVARD_FULL_PATTERN = r"\([A-Za-z][\w\s\-\.]*,\s\d{4},\sp\.\s\d+\)"

# Harvard n.d. citation: (filename.pdf, n.d.)  — our formatter's output
HARVARD_ND_PATTERN = r"\([\w\s\-\.]+,\sn\.d\.\)"


def extract_citations(answer: str) -> list:
    """
    Extract all Harvard-style citations (full or n.d.) from answer.
    Returns a list of matched citation strings.
    """
    full = re.findall(HARVARD_FULL_PATTERN, answer)
    nd = re.findall(HARVARD_ND_PATTERN, answer)
    return full + nd


def has_valid_citations(answer: str) -> bool:
    """
    Returns True if at least one Harvard-style citation exists in the answer.
    """
    return len(extract_citations(answer)) > 0


def count_citations(answer: str) -> int:
    """
    Returns number of citations found in the answer.
    """
    return len(extract_citations(answer))