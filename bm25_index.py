# bm25_index.py
# Minimal BM25 implementation for hybrid retrieval

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, chunks):
        """
        chunks: list of (chunk_text, source_filename)
        """
        self.chunks = chunks

        # tokenize chunks
        self.tokenized_chunks = [
            chunk_text.lower().split()
            for chunk_text, _ in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query, top_k=5):
        """
        Returns indices of best matching chunks
        """
        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return ranked_indices[:top_k]
