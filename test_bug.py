import os
import streamlit as st
from mode_router import detect_mode
from structured_extractor import extract_atomic_facts
from quiz_generator import generate_mcqs_from_facts
from per_output_validator import validate_mcq
from metadata_schema import ChunkMeta

def test():
    chunk = ChunkMeta(
        chunk_id="1234",
        doc_id="d1",
        filename="test.pdf",
        text="The human heart has 4 chambers. This is a fact.",
        page=1
    )
    chunks_store = {chunk.chunk_id: chunk}
    
    facts = extract_atomic_facts(chunk)
    print("Facts:", facts)
    
    unique = {}
    for f in facts:
        key = f["text"]
        if key not in unique:
            unique[key] = f
    facts = list(unique.values())
    
    mcqs = generate_mcqs_from_facts(facts, max_questions=3)
    print("MCQs:", mcqs)
    
    for m in mcqs:
        ok, reason, score = validate_mcq(m, chunks_store)
        print("Validated:", ok, reason, score)

if __name__ == "__main__":
    test()
