# genie/structured_extractor.py
import re
from nltk import sent_tokenize
from typing import List, Dict, Any
import logging

_logger = logging.getLogger(__name__)

# simple heuristics / regex patterns
NUMBER_PATTERN = re.compile(r'\b\d{1,3}(?:[,]\d{3})*(?:\.\d+)?\b')  # numbers like 1,234.56
EQUATION_PATTERN = re.compile(r'R_t|R_0|R0|Q\(|\\\(|=|\\begin|\\end')  # loose match
DEFINITION_KEYWORDS = ["is defined as", "is defined", "is called", "refers to", "is the", "are the", "means"]
METHOD_KEYWORDS = ["we use", "we propose", "we train", "we fit", "we compute"]
RESULT_KEYWORDS = ["results", "we find", "we observe", "demonstrate", "shows", "shown"]

def extract_sentences_from_chunk(chunk_text: str) -> List[str]:
    # Use nltk's sentence tokenizer (ensure nltk punkt is installed)
    try:
        sents = sent_tokenize(chunk_text)
    except Exception:
        # fallback: naive split on periods
        sents = [s.strip() for s in chunk_text.split('.') if s.strip()]
    return sents

def sentence_type(sent: str) -> str:
    s = sent.lower()
    if EQUATION_PATTERN.search(sent):
        return "equation"
    if any(k in s for k in DEFINITION_KEYWORDS):
        return "definition"
    if any(k in s for k in METHOD_KEYWORDS):
        return "method"
    if any(k in s for k in RESULT_KEYWORDS):
        return "result"
    if NUMBER_PATTERN.search(sent):
        return "numeric"
    return "fact"

def extract_atomic_facts(chunk_meta) -> List[Dict[str,Any]]:
    """
    chunk_meta: object with .text, .doc_id, .page, .chunk_id
    returns list of dicts:
    { 'type':..., 'text':..., 'source': {doc,page,chunk_id} }
    """
    out=[]
    sents = extract_sentences_from_chunk(chunk_meta.text)
    for s in sents:
        t = sentence_type(s)
        # conservative: require min length
        if len(s.split()) < 3:
            continue
        # only return sentences that look factual
        if t in ("equation", "definition", "numeric", "fact", "result", "method"):
            out.append({
                "type": t,
                "text": s.strip(),
                "source": {
                    "doc_id": getattr(chunk_meta, "doc_id", None) if not hasattr(chunk_meta, 'get') else chunk_meta.get("doc_id"),
                    "filename": getattr(chunk_meta, "filename", None) if not hasattr(chunk_meta, 'get') else chunk_meta.get("filename"),
                    "page": getattr(chunk_meta, "page", None) if not hasattr(chunk_meta, 'get') else chunk_meta.get("page"),
                    "chunk_id": getattr(chunk_meta, "chunk_id", None) if not hasattr(chunk_meta, 'get') else chunk_meta.get("chunk_id")
                }
            })
    return out
