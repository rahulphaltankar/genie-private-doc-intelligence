# hybrid_retriever.py
# Combines FAISS vector search and BM25 keyword search safely

def hybrid_search(
    query,
    vector_store,
    encoder_model,
    bm25_index,
    chunks,
    top_k=10
):
    """
    Returns hybrid retrieval chunk indices
    """

    # Step 1: Vector search (existing FAISS)
    question_embedding = encoder_model.encode([query])

    D, I = vector_store.search(
        question_embedding.astype('float32'),
        k=top_k
    )

    vector_indices = list(I[0])

    # Step 2: BM25 search
    bm25_indices = bm25_index.search(query, top_k=top_k)

    # Step 3: Merge without duplicates
    merged = []

    for idx in vector_indices + bm25_indices:
        if idx not in merged:
            merged.append(idx)

    print("Hybrid indices:", merged)
    return merged[:top_k]
