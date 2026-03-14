import numpy as np
from genie.pipeline.retriever import hybrid_search
from genie.pipeline.bm25_index import BM25Index
from genie.schemas.metadata_schema_v3 import ChunkMeta

def evaluate_retrieval(queries, ground_truth, chunks, embeddings, bm25_index, top_k=5):
    """
    Very basic precision@k evaluator.
    queries: list of queries
    ground_truth: list of expected chunk_ids for each query
    """
    precisions = []
    
    for i, query in enumerate(queries):
        expected_id = ground_truth[i]
        
        # We need a dummy embedding or mock one for testing if not using a model here
        # For simplicity, we assume this is called within a context where model is available
        # or we mock the search results.
        
        # Here we just mock the interface for illustration
        results = hybrid_search(query, np.zeros(384), embeddings, chunks, bm25_index, top_k=top_k)
        result_ids = [c.chunk_id for c in results]
        
        precision = 1.0 if expected_id in result_ids else 0.0
        precisions.append(precision)
        
    avg_precision = sum(precisions) / len(precisions)
    return avg_precision

if __name__ == "__main__":
    print("Retrieval evaluation module loaded.")
