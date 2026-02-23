# hybrid_retriever.py
# Combines FAISS vector search and BM25 keyword search using dynamic weighting

def hybrid_search(
    query,
    vector_store,
    encoder_model,
    bm25_index,
    chunks,
    top_k=10,
    alpha=0.5, # 1.0 = All Vector, 0.0 = All BM25
    metadata_boost=0.0
):
    """
    Returns hybrid retrieval chunk indices based on weighted scores
    """
    
    # Vector Search (Dense)
    question_embedding = encoder_model.encode([query])
    vec_d, vec_i = vector_store.search(
        question_embedding.astype('float32'),
        k=top_k * 2 # Fetch more for re-ranking pool
    )
    
    # BM25 Search (Sparse)
    bm25_scores = bm25_index.score_all(query)
    
    # Normalize scores (min-max)
    import numpy as np
    
    # Normalize FAISS (L2 distances, lower is better, so invert)
    vec_scores_raw = dict(zip(vec_i[0], vec_d[0]))
    if vec_scores_raw:
        max_vec = max(vec_scores_raw.values())
        min_vec = min(vec_scores_raw.values())
        range_vec = max_vec - min_vec if max_vec > min_vec else 1.0
        vec_scores_norm = {k: 1.0 - ((v - min_vec) / range_vec) for k, v in vec_scores_raw.items()}
    else:
        vec_scores_norm = {}
        
    # Normalize BM25 (higher is better)
    if len(bm25_scores) > 0:
        max_bm25 = max(bm25_scores)
        min_bm25 = min(bm25_scores)
        range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        bm25_scores_norm = [(s - min_bm25) / range_bm25 for s in bm25_scores]
    else:
        bm25_scores_norm = [0.0] * len(chunks)
        
    # Combine scores
    final_scores = {}
    
    # Seed with all BM25 scores (weighted by 1-alpha)
    for i in range(len(chunks)):
        final_scores[i] = (1.0 - alpha) * bm25_scores_norm[i]
        
    # Add FAISS scores (weighted by alpha)
    for idx, score in vec_scores_norm.items():
        if idx != -1: # valid index
            final_scores[idx] += alpha * score
            
    # Apply Metadata Boost for potential headers (e.g., short lines, all caps, or title matches)
    if metadata_boost > 0:
        for i, chunk in enumerate(chunks):
            # Header heuristic: short text (less than 100 chars) or matches title
            is_header = len(chunk.text) < 100 or (hasattr(chunk, 'title') and chunk.title and chunk.title.lower() in chunk.text.lower())
            if is_header:
                final_scores[i] += metadata_boost

    # Sort and take top_k
    sorted_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    
    return sorted_indices[:top_k]
