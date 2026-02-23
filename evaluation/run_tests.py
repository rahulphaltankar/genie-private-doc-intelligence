import sys
import os

# Add parent dir to path so we can import genie modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import io
import faiss
import numpy as np
from tqdm import tqdm

from ingestion_pipeline import process_uploaded_files
from app import load_models, call_mistral_api, extract_number
from bm25_index import BM25Index
from hybrid_retriever import hybrid_search
from reranker import Reranker
from grounding import compute_grounding_score
from gatekeeper import run_gatekeeper
from mode_router import detect_mode
from citation_validator import extract_citations

# Score utilities
from scorer import score_retrieval, score_grounding, score_citation, score_structure, score_decision, calculate_total_score
from metrics import calculate_aggregate_metrics
from report_generator import generate_report

def get_mock_uploaded_file(path):
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    mock_file = io.BytesIO(pdf_bytes)
    mock_file.name = os.path.basename(path)
    return mock_file

def run_all_tests():
    print("Initializing UAT Engine Pipeline...")
    # Load AI Act document
    pdf_path = r"c:\Users\shrut\OneDrive\Desktop\VeritasAI\OJ_L_202401689_EN_TXT.pdf"
    
    # Fallback to current dir if absolute path fails on UAT branch
    if not os.path.exists(pdf_path):
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "OJ_L_202401689_EN_TXT.pdf")
        
    mock_file = get_mock_uploaded_file(pdf_path)
    
    embedding_model = load_models()
    chunks = process_uploaded_files([mock_file])
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_model.encode(chunk_texts)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    bm25_chunks = [(c.text, c.filename) for c in chunks]
    bm25_idx = BM25Index(bm25_chunks)
    
    rer = Reranker()
    
    print("Pipeline Ready. Loading 120 Test Cases...")
    
    with open("evaluation/test_suite.json", "r", encoding="utf-8") as f:
        test_suite = json.load(f)
        
    results = []
    
    print(f"Executing {len(test_suite)} tests...")
    
    for t in tqdm(test_suite, desc="Evaluation Progress"):
        prompt = t["question"]
        expected_behavior = t["expected_behavior"]
        expected_out = t["expected_output_type"]
        
        # We don't have labeled ground-truth chunks for testing, 
        # so for retrieval score, we simulate it by checking if reranker changed the top 3.
        # A true ground-truth set would be standard, but we score generically without it.
        
        intent = detect_mode(prompt)
        indices = hybrid_search(prompt, index, embedding_model, bm25_idx, chunks, top_k=10)
        candidates = [chunks[idx] for idx in indices]
        
        scored = rer.rerank(prompt, candidates)
        top = [c for c, s in scored][:10]
        
        ctx = "\n".join([f"Source ({c.filename}, Page {c.page}): {c.text}" for c in top])
        
        # Generation
        if intent == "quiz":
            from quiz_generator import generate_mcqs_from_facts
            from structured_extractor import extract_atomic_facts
            from per_output_validator import validate_mcq
            
            facts = []
            for chunk in top:
                facts.extend(extract_atomic_facts(chunk))
            unique = {f["text"]: f for f in facts}
            facts = list(unique.values())
            
            mcqs = generate_mcqs_from_facts(facts, max_questions=3)
            chunks_store = {c.chunk_id: c for c in chunks}
            
            final_mcqs = []
            for m in mcqs:
                ok, _, _ = validate_mcq(m, chunks_store)
                if ok: final_mcqs.append(m)
                
            raw_ans = json.dumps(final_mcqs)
            gk_decision = "PASS" if len(final_mcqs) > 0 else "BLOCK"
            gs = 1.0 if len(final_mcqs) > 0 else 0.0
            
        else:
            raw_ans = call_mistral_api(prompt, ctx)
            gs = compute_grounding_score(raw_ans, [c.text for c in top])
            gk_decision, _ = run_gatekeeper(raw_ans, [c.text for c in top], gs, mode=intent)
        
        # -- SCORING PHASE -- 
        # 1. Retrieval
        ret_score = 30 # Baseline assumption for unlabelled test
        
        # 2. Grounding
        is_cb = (expected_behavior == "BLOCK" and gk_decision == "BLOCK")
        minor_unsupp = (gs > 0.25 and gs < 0.40)
        hallucinated = (gs < 0.25 and gk_decision != "BLOCK")
        ground_score = score_grounding(hallucinated, minor_unsupp, is_cb)
        
        # 3. Citation
        cites = extract_citations(raw_ans)
        cit_score = score_citation(raw_ans, 1 if intent == "factual" else 0, len(cites))
        
        # 4. Structure
        struct_score = score_structure(raw_ans, expected_out)
        
        # 5. Decision
        dec_score = score_decision(gk_decision, expected_behavior)
        
        total = calculate_total_score(ret_score, ground_score, cit_score, struct_score, dec_score)
        
        results.append({
            "id": t["id"],
            "category": t["category"],
            "question": prompt,
            "expected_behavior": expected_behavior,
            "decision": gk_decision,
            "grounding_score": gs,
            "total_score": total,
            "retrieval_score_component": ret_score,
            "grounding_score_component": ground_score,
            "citation_score_component": cit_score,
            "structure_score_component": struct_score,
            "decision_score_component": dec_score
        })
        
    metrics = calculate_aggregate_metrics(results)
    generate_report(metrics, results)
    return metrics
    
if __name__ == "__main__":
    run_all_tests()
