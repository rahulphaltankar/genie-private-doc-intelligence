import json
import os
import sys

def run_ci_gate():
    report_path = "evaluation/reports/latest_report.json"
    
    if not os.path.exists(report_path):
        print(f"[ERROR] Report not found at {report_path}. Run run_tests.py first.")
        sys.exit(1)
        
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    metrics = report.get("metrics", {})
    
    # Extract
    avg_score = metrics.get("average_score", 0)
    block_acc = metrics.get("block_accuracy", 0)
    hallucination_rate = metrics.get("hallucination_rate", 1.0)
    recall_at_3 = metrics.get("retrieval_recall_at_3", 0)
    
    # Evaluate Rules
    print("--- 🚦 CI GATE EVALUATION ---")
    fail = False
    
    if avg_score < 80:
        print(f"[FAIL] Average Score: {avg_score:.2f} < 80")
        fail = True
    else:
        print(f"[PASS] Average Score: {avg_score:.2f} >= 80")

    if block_acc < 0.95:
        print(f"[FAIL] Block Accuracy: {block_acc*100:.1f}% < 95%")
        fail = True
    else:
        print(f"[PASS] Block Accuracy: {block_acc*100:.1f}% >= 95%")
        
    if hallucination_rate > 0.02:
        print(f"[FAIL] Hallucination Rate: {hallucination_rate*100:.1f}% > 2.0%")
        fail = True
    else:
        print(f"[PASS] Hallucination Rate: {hallucination_rate*100:.1f}% <= 2.0%")
        
    if recall_at_3 < 0.80:
        print(f"[FAIL] Recall@3: {recall_at_3*100:.1f}% < 80%")
        fail = True
    else:
        print(f"[PASS] Recall@3: {recall_at_3*100:.1f}% >= 80%")
        
    if fail:
        print("\n[RESULT] ❌ CI GATE FAILED. Golden suite regression detected.")
        sys.exit(1)
    else:
        print("\n[RESULT] ✅ CI GATE PASSED. Ready for deployment.")
        sys.exit(0)

if __name__ == "__main__":
    run_ci_gate()
