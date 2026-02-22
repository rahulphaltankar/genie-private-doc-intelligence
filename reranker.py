# genie/reranker.py
from sentence_transformers import CrossEncoder
import numpy as np
import logging

# Choose a lightweight cross-encoder for accuracy but reasonable speed.
# "cross-encoder/ms-marco-MiniLM-L-6-v2" is a good tradeoff.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name=MODEL_NAME):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            _logger.warning(f"Failed to load CrossEncoder {model_name}: {e}")
            self.model = None

    def rerank(self, query, candidate_chunks):
        """
        candidate_chunks: list of ChunkMeta objects or dicts with .text or ['text']
        returns: list of candidates sorted by score desc: [(chunk, score), ...]
        """
        if not candidate_chunks:
            return []

        texts = [c.text if hasattr(c, "text") else c["text"] for c in candidate_chunks]
        pairs = [[query, t] for t in texts]

        if self.model:
            scores = self.model.predict(pairs)  # higher is better
            scored = list(zip(candidate_chunks, scores))
            scored.sort(key=lambda x: -x[1])
            return scored
        else:
            # Fallback: use simple sentence-transformers cosine similarity (bi-encoder)
            from sentence_transformers import SentenceTransformer, util
            emb_model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = emb_model.encode(query, convert_to_tensor=True)
            t_embs = emb_model.encode(texts, convert_to_tensor=True)
            cos_scores = util.cos_sim(q_emb, t_embs).cpu().numpy()[0]
            scored = list(zip(candidate_chunks, cos_scores))
            scored.sort(key=lambda x: -x[1])
            return scored
