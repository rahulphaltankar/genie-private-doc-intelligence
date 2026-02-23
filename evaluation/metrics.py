def calculate_aggregate_metrics(results):
    total_score = 0
    total_tests = len(results)
    
    if total_tests == 0:
        return {}

    hallucination_count = 0
    correct_block_count = 0
    expected_block_count = 0
    retrieval_at_3_count = 0
    retrieval_applicable_count = 0

    for r in results:
        total_score += r["total_score"]
        
        # Block Accuracy
        if r["expected_behavior"] == "BLOCK":
            expected_block_count += 1
            if r["decision"] == "BLOCK":
                correct_block_count += 1
                
        # Hallucination Rate (Proxy: Passed gatekeeper without valid citations, or generated false claims)
        # Using scorer logic: if grounding score == 0 (hallucinated claim)
        if r["grounding_score_component"] == 0 and r["decision"] != "BLOCK":
            hallucination_count += 1
            
        # Retrieval Recall at 3 (Only applicable if we expect a document retrieval match, which is everything not an adversarial trap)
        if r["expected_behavior"] != "BLOCK":
            retrieval_applicable_count += 1
            if r["retrieval_score_component"] == 30:
                retrieval_at_3_count += 1

    average_score = total_score / total_tests
    hallucination_rate = hallucination_count / total_tests
    
    block_accuracy = 1.0
    if expected_block_count > 0:
        block_accuracy = correct_block_count / expected_block_count
        
    retrieval_recall_at_3 = 1.0
    if retrieval_applicable_count > 0:
        retrieval_recall_at_3 = retrieval_at_3_count / retrieval_applicable_count

    return {
        "total_tests": total_tests,
        "average_score": average_score,
        "hallucination_rate": hallucination_rate,
        "block_accuracy": block_accuracy,
        "retrieval_recall_at_3": retrieval_recall_at_3
    }
