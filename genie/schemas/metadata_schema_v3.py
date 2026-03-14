from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ChunkMeta:
    chunk_id: str            # uuid
    doc_id: str              # canonical doc identifier
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    text: str = ""
    tokens: int = 0
    created_at: str = datetime.utcnow().isoformat()
    permissions: Optional[List[str]] = None
