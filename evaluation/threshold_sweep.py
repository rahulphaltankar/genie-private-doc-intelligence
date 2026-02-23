import sys
import os
import json
import re

# Add parent dir to path so we can import genie modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gatekeeper
from run_tests import run_all_tests

def run_sweep():
    print("Starting Threshold Sweep testing...")
    
    # Identify current threshold directly from gatekeeper logic
    current_threshold = gatekeeper.FACTUAL_PASS_THRESHOLD
    
    thresholds_to_test = [
        round(current_threshold - 0.05, 2),
        current_threshold,
        round(current_threshold + 0.05, 2),
        round(current_threshold + 0.10, 2)
    ]
    
    sweep_results = []
    
    for t in thresholds_to_test:
        print(f"\n=========================================")
        print(f"Running evaluation with Threshold = {t:.2f}")
        print(f"=========================================")
        
        # Override the gatekeeper threshold temporarily
        gatekeeper.FACTUAL_PASS_THRESHOLD = t
        gatekeeper.SYNTHESIS_PASS_THRESHOLD = max(0.20, t - 0.15)
        
        # Run tests and capture metrics
        metrics = run_all_tests()
        
        sweep_results.append({
            "threshold": t,
            "metrics": metrics
        })
        
    print("\n\n=== SWEEP RESULTS ===")
    
    for r in sweep_results:
        m = r["metrics"]
        avg = m.get("average_score", 0)
        hr = m.get("hallucination_rate", 1.0)
        ba = m.get("block_accuracy", 0.0)
        rec = m.get("retrieval_recall_at_3", 0.0)
        
        print(f"Threshold: {r['threshold']:.2f} | Avg Score: {avg:.1f} | HR: {hr*100:.1f}% | Block Acc: {ba*100:.1f}%")
        
    out_path = "evaluation/reports/threshold_sweep.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)
        
    print(f"\nSaved sweep report to {out_path}")
    
    # 🧠 SELECTION CRITERIA
    valid_candidates = [
        r for r in sweep_results 
        if r["metrics"]["block_accuracy"] >= 0.95 
        and r["metrics"]["average_score"] >= 80
    ]
    
    if valid_candidates:
        # Sort by lowest hallucination rate, then higher threshold (safer) if there is a tradeoff
        valid_candidates.sort(key=lambda x: (x["metrics"]["hallucination_rate"], -x["threshold"]))
        best_threshold = valid_candidates[0]["threshold"]
        print(f"\n[WINNER] Best threshold selected: {best_threshold:.2f}")
    else:
        best_threshold = current_threshold
        print(f"\n[WINNER] No threshold met all criteria! Sticking with current {best_threshold:.2f}")
        
    # Write the best threshold permanently back to the codebase
    gk_path = os.path.join(os.path.dirname(__file__), '..', 'gatekeeper.py')
    with open(gk_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    content = re.sub(
        r"FACTUAL_PASS_THRESHOLD = [0-9.]+", 
        f"FACTUAL_PASS_THRESHOLD = {best_threshold:.2f}", 
        content
    )
    content = re.sub(
        r"SYNTHESIS_PASS_THRESHOLD = [0-9.]+", 
        f"SYNTHESIS_PASS_THRESHOLD = {max(0.20, best_threshold - 0.15):.2f}", 
        content
    )
    
    with open(gk_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\n[SUCCESS] Updated production gatekeeper.py with optimal threshold: {best_threshold:.2f}")

if __name__ == "__main__":
    run_sweep()
