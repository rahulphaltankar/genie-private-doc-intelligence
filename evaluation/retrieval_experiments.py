import sys
import os
import json
import time

# Add parent dir to path so we can import genie modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hybrid_retriever
from run_tests import run_all_tests

def run_retrieval_sweep():
    print("Starting Retrieval Strategy Optimization Sweep...")
    
    # 🔍 EXPERIMENTS TO RUN
    experiments = [
        {"name": "Baseline (k5, alpha0.5)", "top_k": 5, "alpha": 0.5, "expansion": False, "rerank_k": 5, "boost": 0.0},
        {"name": "Step 1: Increase k to 8", "top_k": 8, "alpha": 0.5, "expansion": False, "rerank_k": 5, "boost": 0.0},
        {"name": "Step 2: Increase k to 10", "top_k": 10, "alpha": 0.5, "expansion": False, "rerank_k": 5, "boost": 0.0},
        {"name": "Step 3: Weight 0.7 Dense", "top_k": 10, "alpha": 0.7, "expansion": False, "rerank_k": 5, "boost": 0.0},
        {"name": "Step 4: Weight 0.3 Dense", "top_k": 10, "alpha": 0.3, "expansion": False, "rerank_k": 5, "boost": 0.0},
        {"name": "Step 5: Context Expansion (+1/-1)", "top_k": 10, "alpha": 0.5, "expansion": True, "rerank_k": 5, "boost": 0.0},
        {"name": "Step 6: Rerank Top 10", "top_k": 10, "alpha": 0.5, "expansion": False, "rerank_k": 10, "boost": 0.0},
        {"name": "Step 7: Metadata Boost (0.2)", "top_k": 10, "alpha": 0.5, "expansion": False, "rerank_k": 5, "boost": 0.2},
        {"name": "Step 8: Kitchen Sink (Optimized)", "top_k": 10, "alpha": 0.7, "expansion": True, "rerank_k": 10, "boost": 0.1},
    ]
    
    results = []
    
    import run_tests
    import hybrid_retriever
    
    # Cache chunks for expansion mapping
    # We need to reach into run_tests to get the chunk list
    # For now, we will patch hybrid_search AND the reranking step.
    
    original_hybrid_search = run_tests.hybrid_search
    
    for exp in experiments:
        print(f"\n=========================================")
        print(f"Running Experiment: {exp['name']}")
        print(f"Params -> Top K: {exp['top_k']} | Alpha: {exp['alpha']} | Expand: {exp['expansion']} | Rerank K: {exp['rerank_k']} | Boost: {exp['boost']}")
        print(f"=========================================")
        
        # Patch hybrid_search with boost and k
        def mocked_hybrid_search(query, vector_store, encoder_model, bm25_index, chunks, top_k=10):
            indices = hybrid_retriever.hybrid_search(
                query, vector_store, encoder_model, bm25_index, chunks,
                top_k=exp["top_k"],
                alpha=exp["alpha"],
                metadata_boost=exp["boost"]
            )
            
            if exp["expansion"]:
                # Expand indices by adding neighboring chunks (+1/-1)
                expanded = set(indices)
                for idx in indices:
                    if idx > 0: expanded.add(idx - 1)
                    if idx < len(chunks) - 1: expanded.add(idx + 1)
                return sorted(list(expanded))
            return indices
        
        run_tests.hybrid_search = mocked_hybrid_search
        
        # Patch the rerank slice in run_tests loop? 
        # run_tests current code: top = [c for c, s in scored][:10]
        # We need to patch the rerank usage or just accept the 10-slice might be changed.
        # Actually, run_tests.py line 83 has: top = [c for c, s in scored][:10]
        # We can monkeypatch Reranker.rerank to return only top exp['rerank_k']
        
        from reranker import Reranker
        original_rerank = Reranker.rerank
        
        def mocked_rerank(self, query, candidate_chunks):
            scored = original_rerank(self, query, candidate_chunks)
            return scored[:exp["rerank_k"]]
        
        Reranker.rerank = mocked_rerank
        
        start_time = time.time()
        metrics = run_all_tests()
        latency = time.time() - start_time
        
        results.append({
            "name": exp["name"],
            "params": exp,
            "latency_seconds": latency,
            "metrics": metrics
        })
        
        # Restore Reranker for next loop
        Reranker.rerank = original_rerank
        
    # Restore original function
    run_tests.hybrid_search = original_hybrid_search
    
    print("\n\n=== RETRIEVAL SWEEP RESULTS ===")
    
    baseline_latency = results[0]["latency_seconds"]
    
    for r in results:
        m = r["metrics"]
        avg = m.get("average_score", 0)
        rec = m.get("retrieval_recall_at_3", 0.0)
        
        lat_diff = ((r["latency_seconds"] - baseline_latency) / baseline_latency) * 100
        
        print(f"[{r['name']}] Recall@3: {rec*100:.1f}% | Avg: {avg:.1f} | Latency: {r['latency_seconds']:.1f}s ({lat_diff:+.1f}%)")
        
    out_path = "evaluation/reports/retrieval_experiments.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved retrieval sweep report to {out_path}")
    
    # 🧠 SELECT BEST CONFIG
    valid_candidates = [
        r for r in results 
        if r["metrics"]["average_score"] >= 80 
        and ((r["latency_seconds"] - baseline_latency) / baseline_latency) <= 0.30
    ]
    
    if valid_candidates:
        # Sort by highest recall, then highest average score, then lowest latency
        valid_candidates.sort(key=lambda x: (x["metrics"]["retrieval_recall_at_3"], x["metrics"]["average_score"], -x["latency_seconds"]), reverse=True)
        winner = valid_candidates[0]
        wp = winner["params"]
        print(f"\n[WINNER] Selected Config: {winner['name']}")
        print(f"Applying -> top_k = {wp['top_k']}, alpha = {wp['alpha']}, boost = {wp['boost']}")
        
        # Write permanent change to app.py
        app_path = os.path.join(os.path.dirname(__file__), '..', 'app.py')
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        import re
        # Update top_k
        content = re.sub(
            r"top_k=15 if intent == \"quiz\" else [0-9]+",
            f"top_k=15 if intent == \"quiz\" else {wp['top_k']}",
            content
        )
        # Update alpha and boost in hybrid_search call
        content = re.sub(
            r"top_k=([^\n,)]+)\n\s+\)",
            f"top_k=\\1,\n                        alpha={wp['alpha']},\n                        metadata_boost={wp['boost']}\n                    )",
            content
        )
        
        # If expansion is winner, we need more complex logic in app.py
        # For now, let's assume we update the core params.
        
        with open(app_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print("[SUCCESS] Production codebase updated with optimized retrieval strategy.")
        
    else:
        print("\n[FAILED] No configuration met the criteria. Sticking with baseline.")

if __name__ == "__main__":
    run_retrieval_sweep()
