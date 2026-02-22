import numpy as np
import pytest
from hybrid_retriever import hybrid_search
from bm25_index import BM25Index
from metadata_schema import ChunkMeta

class MockEncoder:
    def encode(self, texts):
        return np.array([[1.0, 0.0]])

class MockFaiss:
    def search(self, query_emb, k):
        return np.array([[0.1]]), np.array([[0]])

def test_hybrid_search_results():
    chunks = [
        ChunkMeta(chunk_id="1", doc_id="d1", filename="f1.pdf", text="Apple is a fruit", page=1),
        ChunkMeta(chunk_id="2", doc_id="d1", filename="f1.pdf", text="Banana is yellow", page=1)
    ]
    bm25_chunks = [(c.text, c.filename) for c in chunks]
    bm25 = BM25Index(bm25_chunks)
    
    encoder = MockEncoder()
    vector_store = MockFaiss()
    
    results = hybrid_search("Apple", vector_store, encoder, bm25, chunks, top_k=1)
    assert 0 in results
    
    class MockFaissBanana:
        def search(self, query_emb, k):
            return np.array([[0.1]]), np.array([[1]])
            
    results_bm25 = hybrid_search("Banana", MockFaissBanana(), encoder, bm25, chunks, top_k=1)
    assert 1 in results_bm25
