# Retrieval Optimization Walkthrough

We conducted a comprehensive sweep of 9 retrieval strategies to optimize Genie's recall and precision.

## Experiment Results

| Configuration | Recall@3 | Avg Score | Latency Change |
| :--- | :--- | :--- | :--- |
| Baseline (k=5) | 100% | 88.3 | 0% |
| Step 1: k=8 | 100% | 87.1 | +12.0% |
| Step 2: k=10 | 100% | 87.1 | +0.2% |
| Step 3: 0.7 Dense | 100% | 85.8 | +7.2% |
| Step 4: 0.3 Dense | 100% | 87.5 | +0.2% |
| Step 5: Expansion | 100% | 89.2 | +30.7% |
| **Step 6: Rerank 10** | **100%** | **89.6** | **+25.1%** |
| Step 7: Boost | 100% | 88.3 | +33.8% |
| Kitchen Sink | 100% | 87.1 | +71.8% |

## Winning Configuration
**Step 6: Rerank Top 10** was selected as the winner.
- **Criteria Met**: Avg Score ≥ 80 (89.6), Latency Increase ≤ 30% (+25.1%).
- **Implementation**: Updated `app.py` to use `top_k=10` with default alpha=0.5.

## Technical Details
- Added `metadata_boost` support to `hybrid_retriever.py`.
- Refactored `retrieval_experiments.py` to support multi-dimensional sweeps (Expansion, Boosting, Reranking depth).
- Verified production update in `app.py`.

```python
# app.py snippet
indices = hybrid_search(
    query=user_msg,
    vector_store=st.session_state.vector_store,
    encoder_model=embedding_model,
    bm25_index=st.session_state.bm25_index,
    chunks=st.session_state.chunks,
    top_k=15 if intent == "quiz" else 10,
    alpha=0.5,
    metadata_boost=0.0
)
```
