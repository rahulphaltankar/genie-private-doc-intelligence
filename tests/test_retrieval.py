import numpy as np
import pytest
from hybrid_retriever import hybrid_search
from bm25_index import BM25Index
from metadata_schema import ChunkMeta

def test_hybrid_search_results():
    chunks = [
        ChunkMeta(chunk_id="1", doc_id="d1", filename="f1.pdf", text="Apple is a fruit", page=1),
        ChunkMeta(chunk_id="2", doc_id="d1", filename="f1.pdf", text="Banana is yellow", page=1)
    ]
    chunk_texts = [c.text for c in chunks]
    bm25 = BM25Index(chunk_texts)
    
    # Mock embeddings
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    q_emb = np.array([1.0, 0.0])
    
    results = hybrid_search("Apple", q_emb, embs, chunks, bm25, alpha=1.0, top_k=1)
    assert results[0].chunk_id == "1"
    
    results_bm25 = hybrid_search("Banana", q_emb, embs, chunks, bm25, alpha=0.0, top_k=1)
    assert results_bm25[0].chunk_id == "2"
