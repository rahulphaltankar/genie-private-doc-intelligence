from sentence_transformers import SentenceTransformer, util

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_grounding_score(answer: str, retrieved_chunks: list) -> float:
    """
    Computes grounding score between answer and retrieved chunks.
    """

    if not answer or not retrieved_chunks:
        return 0.0

    answer_embedding = model.encode(answer, convert_to_tensor=True)
    chunk_embeddings = model.encode(retrieved_chunks, convert_to_tensor=True)

    similarities = util.cos_sim(answer_embedding, chunk_embeddings)

    max_score = similarities.max().item()

    return round(max_score, 3)