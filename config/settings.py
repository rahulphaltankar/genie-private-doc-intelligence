"""
config/settings.py — Genie v4 Application Settings

All secrets loaded from .env (or GCP Secret Manager in production).
Never reference this module from v3 code — this is the v4 config contract.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # GCP / Cloud
    GCP_PROJECT_ID: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""  # dev fallback only — not in production routing

    # Database
    PGVECTOR_CONNECTION_STRING: str = ""

    # Grounding thresholds (Dual-Lock gates)
    SIMILARITY_THRESHOLD: float = 0.35
    PARTIAL_THRESHOLD: float = 0.20

    # Retrieval
    RETRIEVAL_TOP_K: int = 20
    ALPHA: float = 0.5          # 1.0 = pure vector, 0.0 = pure BM25
    CONTEXT_CHUNKS: int = 5

    # Chunking
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # Observability
    TRACE_LOG_PATH: str = "genie_trace.jsonl"

    # BM25 cache
    BM25_CACHE_PATH: str = "bm25_index.pkl"

    # Vertex AI
    EMBEDDING_LOCATION: str = "us-central1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
