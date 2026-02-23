import json
import os

def generate_report(metrics, detail_results, output_path="evaluation/reports/latest_report.json"):
    
    # Ensure reports directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = {
        "metrics": metrics,
        "results": detail_results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[REPORT GENERATED] Saved to {output_path}")
    
    print("\n--- TEST EXECUTION SUMMARY ---")
    print(f"Total Tests Run: {metrics['total_tests']}")
    print(f"Average Score:   {metrics['average_score']:.2f} / 100")
    print(f"Block Accuracy:  {metrics['block_accuracy']*100:.1f}%")
    print(f"Recall@3:        {metrics['retrieval_recall_at_3']*100:.1f}%")
    print(f"Hallucination Rate: {metrics['hallucination_rate']*100:.1f}%")
    
    # Print lowest 10 scoring
    sorted_res = sorted(detail_results, key=lambda x: x["total_score"])
    print("\n[!] Lowest Scoring Questions:")
    for i, r in enumerate(sorted_res[:5]):
        print(f" {i+1}. Q{r['id']} ({r['category']}) - Score: {r['total_score']} | Decision: {r['decision']}")
