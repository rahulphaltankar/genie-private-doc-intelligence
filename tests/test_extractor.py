import pytest

def test_extract_basic():
    from genie.tools.structured_extractor import extract_atomic_facts
    chunk = type("C", (), {})()
    chunk.text = "The Transformer uses multi-head attention. We train models for 10 epochs. R_t is defined as beta/gamma. Irrelevant."
    chunk.doc_id = "doc1"
    chunk.chunk_id = "c1"
    chunk.page = 1
    f = extract_atomic_facts(chunk)
    assert any("multi-head attention" in item["text"] for item in f)
    assert any("10 epochs" in item["text"] for item in f)
    assert any("R_t" in item["text"] for item in f)
