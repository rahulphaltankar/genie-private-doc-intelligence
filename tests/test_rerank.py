import pytest

def test_rerank_basic():
    from reranker import Reranker
    r = Reranker()
    q = "What is multi-head attention?"
    candidates = [{"text":"The Transformer uses multi-head attention to attend."},
                  {"text":"This is unrelated text about cooking."}]
    scored = r.rerank(q, candidates)
    assert scored[0][1] >= scored[1][1]
