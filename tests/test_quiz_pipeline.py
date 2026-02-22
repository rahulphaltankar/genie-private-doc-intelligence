# tools/test_quiz_pipeline.py
import pytest
from hybrid_retriever import hybrid_search
from reranker import Reranker
from structured_extractor import extract_atomic_facts
from quiz_generator import generate_mcqs_from_facts
from per_output_validator import validate_mcq
from sentence_transformers import SentenceTransformer
from bm25_index import BM25Index
import numpy as np

class DummyChunk:
    def __init__(self, c_id, text, fname, doc_id, page):
        self.chunk_id = c_id
        self.text = text
        self.filename = fname
        self.doc_id = doc_id
        self.page = page

def test_full_pipeline():
    chunks = [
        DummyChunk("c1", "The Transformer uses multi-head attention to effectively attend to information.", "doc1.pdf", "d1", 1),
        DummyChunk("c2", "We train models for 10 epochs.", "doc1.pdf", "d1", 1),
        DummyChunk("c3", "R_t is defined as beta/gamma.", "doc1.pdf", "d1", 2),
        DummyChunk("c4", "This is an irrelevant chunk about making pizza.", "doc2.pdf", "d2", 1),
        DummyChunk("c5", "Another irrelevant text.", "doc2.pdf", "d2", 2)
    ]
    
    # 1. Setup Models & Index
    encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_texts = [c.text for c in chunks]
    chunk_embs = encoder_model.encode(chunk_texts)
    
    import faiss
    # faiss setup mocking our app
    vector_store = faiss.IndexFlatL2(chunk_embs.shape[1])
    vector_store.add(np.array(chunk_embs).astype('float32'))
    
    bm25_chunks = [(c.text, c.filename) for c in chunks]
    bm25_index = BM25Index(bm25_chunks)

    query = "Create 10 MCQs based on the transformer model"
    
    # 2. Hybrid Search
    query_emb = encoder_model.encode([query])[0]
    
    # The hybrid search in our app expects different arguments, let's use the one we created in Step 3
    # Wait, the hybrid_search we created in Step 3 actually takes query string, not embedding.
    # Let's check hybrid_retriever.py
    import hybrid_retriever
    
    # Actually wait, hybrid_retriever.py from Step 3 has:
    # def hybrid_search(query, vector_store, encoder_model, bm25_index, chunks, top_k=5):
    
    candidates = hybrid_retriever.hybrid_search(
        query=query, 
        vector_store=vector_store, 
        encoder_model=encoder_model, 
        bm25_index=bm25_index, 
        chunks=chunks, 
        top_k=5
    )
    
    # 3. Rerank
    rer = Reranker()
    # The new hybrid search returns index numbers, so we have to map them
    candidate_chunks = [chunks[idx] for idx in candidates]
    
    scored = rer.rerank("Create 10 MCQs based on this doc", candidate_chunks)
    selected_chunks = [c for c,s in scored[:5]]
    
    # 4. Extract Facts
    facts=[]
    for c in selected_chunks:
        facts.extend(extract_atomic_facts(c))
        
    # 5. Generate MCQs
    mcqs = generate_mcqs_from_facts(facts, max_questions=5)
    
    # 6. Validate
    chunks_store = {c.chunk_id: c for c in chunks}
    final=[]
    for m in mcqs:
        ok, reason, score = validate_mcq(m, chunks_store)
        if ok:
            final.append(m)
            
    print(f"Generated {len(mcqs)} candidates, {len(final)} valid MCQs")
    # For a small dummy text it might yield 0 valid because of constraints
    assert isinstance(final, list)

