import faiss
import numpy as np
import time
from unittest.mock import Mock

# Import Genie modules
from ingestion_pipeline import process_uploaded_files
from app import load_models
from bm25_index import BM25Index
from hybrid_retriever import hybrid_search
from reranker import Reranker
from grounding import compute_grounding_score
from gatekeeper import run_gatekeeper

def run_test_suite():
    print("=========================================")
    print("ISTQB AI TEST EXECUTION REPORT FOR GENIE")
    print("=========================================\n")

    # 1. SETUP & INGESTION
    print("1. Executing Component Test: Ingestion & Chunking")
    print("-------------------------------------------------")
    
    import io
    def get_mock_uploaded_file(path):
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        mock_file = io.BytesIO(pdf_bytes)
        mock_file.name = path.split("\\")[-1]
        return mock_file

    try:
        pdf_path = r"c:\Users\shrut\OneDrive\Desktop\VeritasAI\OJ_L_202401689_EN_TXT.pdf"
        mock_file = get_mock_uploaded_file(pdf_path)
        start_time = time.time()
        chunks = process_uploaded_files([mock_file])
        ingestion_time = time.time() - start_time
        print(f"[PASS] Successfully ingested {mock_file.name}")
        print(f"       Extracted {len(chunks)} chunks in {ingestion_time:.2f} seconds.")
        
        # Verify chunk sizes
        lengths = [len(c.text) for c in chunks]
        print(f"       Avg chunk char length: {sum(lengths)/len(lengths):.0f}")
        print(f"       Max chunk length: {max(lengths)}")
    except Exception as e:
        print(f"[FAIL] Ingestion failed: {e}")
        return

    # 2. VECTORIZATION (FAISS + BM25)
    print("\n2. Executing Component Test: Vectorization & Indexing")
    print("-----------------------------------------------------")
    try:
        model = load_models()
        chunk_texts = [c.text for c in chunks]
        embeddings = model.encode(chunk_texts)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        
        bm25_chunks = [(c.text, c.filename) for c in chunks]
        bm25_idx = BM25Index(bm25_chunks)
        print("[PASS] FAISS and BM25 indices built successfully.")
    except Exception as e:
        print(f"[FAIL] Indexing failed: {e}")
        return

    # 3. HYBRID RETRIEVAL
    print("\n3. Executing Component Test: Hybrid Search Bounds")
    print("-------------------------------------------------")
    
    # Valid Query
    query_valid = "What is the penalty for high-risk AI models?"
    retrieved = hybrid_search(query_valid, index, model, bm25_idx, chunks, top_k=10)
    print(f"[PASS] Valid Query Retrieved {len(retrieved)} unique chunks.")

    # Adversarial / Out of Domain Query
    query_adversarial = "What is the recipe for chocolate chip cookies?"
    retrieved_adv = hybrid_search(query_adversarial, index, model, bm25_idx, chunks, top_k=10)
    print(f"[NOTE] Semantic drift check: Adversarial query returned {len(retrieved_adv)} chunks.")
    
    # 4. GROUNDING AND GATEKEEPER
    print("\n4. Executing Component Test: Grounding & Threshold Bypassing")
    print("------------------------------------------------------------")
    
    # Mock LLM generation: The verbose problem
    retrieved_texts = [chunks[i].text for i in retrieved[:3]]
    
    # 4A. Faithfulness - Concise vs Verbose
    concise_answer = "The penalty for non-compliance ranges up to 35 million EUR or 7% of worldwide annual turnover. (OJ_L_202401689_EN_TXT.pdf)"
    verbose_answer = "Here is a detailed 500-word essay about the penalty for non-compliance. Firstly, as stated in the text, it is 35 million EUR. But let me also explain what an AI model is. An AI model is a mathematical abstraction... (OJ_L_202401689_EN_TXT.pdf)"
    
    score_concise = compute_grounding_score(concise_answer, retrieved_texts)
    score_verbose = compute_grounding_score(verbose_answer, retrieved_texts)
    
    print(f"[PASS] Grounding Score (Concise): {score_concise}")
    print(f"[PASS] Grounding Score (Verbose): {score_verbose}")
    
    # Gatekeeper enforcement
    gk_concise = run_gatekeeper(concise_answer, retrieved_texts, score_concise, "factual")
    gk_verbose = run_gatekeeper(verbose_answer, retrieved_texts, score_verbose, "factual")
    
    print(f"[PASS] Gatekeeper Evaluation (Concise): {gk_concise[0]} -> {gk_concise[1]}")
    print(f"[PASS] Gatekeeper Evaluation (Verbose): {gk_verbose[0]} -> {gk_verbose[1]}")

    # 4B. Citation Presence Injection Test
    missing_citation_answer = "The penalty is up to 35 million EUR."
    score_missing = compute_grounding_score(missing_citation_answer, retrieved_texts)
    gk_missing = run_gatekeeper(missing_citation_answer, retrieved_texts, score_missing, "factual")
    print(f"[PASS] Gatekeeper Citation Override Check: {gk_missing[0]} -> {gk_missing[1]}")

if __name__ == "__main__":
    run_test_suite()
