# tools/test_quiz_pipeline.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hybrid_retriever import hybrid_search
from reranker import Reranker
from structured_extractor import extract_atomic_facts
from quiz_generator import generate_mcqs_from_facts
from per_output_validator import validate_mcq
from metadata_schema import ChunkMeta

def run_test():
    # Mock chunks
    chunks = [
        ChunkMeta(chunk_id="c1", doc_id="d1", filename="f1", text="The Transformer uses multi-head attention. We train models for 10 epochs. R_t is defined as beta/gamma.", page=1),
        ChunkMeta(chunk_id="c2", doc_id="d1", filename="f1", text="Another chunk with padding.", page=1)
    ]
    
    query = "Create 10 MCQs"
    
    # Mock hybrid
    candidates = [0, 1]
    
    rer = Reranker()
    scored = rer.rerank("Create 10 MCQs based on this doc", [chunks[i] for i in candidates])
    selected_chunks = [c for c,s in scored[:10]]
    
    facts=[]
    for c in selected_chunks:
        facts.extend(extract_atomic_facts(c))
        
    mcqs = generate_mcqs_from_facts(facts, max_questions=20)
    
    chunks_store = {c.chunk_id: c for c in chunks}
    final=[]
    for m in mcqs:
        ok, reason, score = validate_mcq(m, chunks_store)
        if ok:
            final.append(m)
            
    print(f"Generated {len(mcqs)} candidates, {len(final)} valid MCQs")
    for f in final:
        print(f"Q: {f['question']}\nA: {f['answer']}")

if __name__ == "__main__":
    run_test()
