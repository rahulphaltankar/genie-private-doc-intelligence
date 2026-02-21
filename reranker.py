from sentence_transformers import CrossEncoder

# Using a standard well-performing cross-encoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidate_chunks):
    """
    Reranks candidate chunks based on query relevance using a cross-encoder.
    """
    if not candidate_chunks:
        return []
        
    pairs = [(query, c.text) for c in candidate_chunks]
    scores = model.predict(pairs)
    
    # Pair each chunk with its score and sort
    ranked_chunks = [c for _, c in sorted(zip(scores, candidate_chunks), key=lambda x: x[0], reverse=True)]
    
    # Store reranker scores if needed for audit (optional enhancement)
    return ranked_chunks
