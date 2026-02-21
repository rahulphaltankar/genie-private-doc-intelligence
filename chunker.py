import uuid
from datetime import datetime
from typing import List
import nltk
from metadata_schema import ChunkMeta

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def chunk_by_page_text(doc_id: str, filename: str, page_texts: List[str], title: str = None, author: str = None) -> List[ChunkMeta]:
    """
    Chunks text by page while preserving page numbers and using semantic sentence splitting.
    """
    chunks = []
    for page_num, text in enumerate(page_texts, start=1):
        if not text.strip():
            continue
            
        sentences = nltk.sent_tokenize(text)
        current_chunk_sentences = []
        current_length = 0
        
        # Approx character threshold for ~300-600 tokens
        # Assuming avg token length ~5 chars, 1200-1500 chars is ~250-300 tokens
        CHAR_THRESHOLD = 1500 
        
        for s in sentences:
            current_chunk_sentences.append(s)
            current_length += len(s)
            
            if current_length > CHAR_THRESHOLD:
                chunk_text = " ".join(current_chunk_sentences)
                c = ChunkMeta(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=filename,
                    title=title,
                    author=author,
                    page=page_num,
                    text=chunk_text,
                    tokens=len(chunk_text.split()),
                    created_at=now_iso()
                )
                chunks.append(c)
                
                # Overlap: keep last 2 sentences for context continuity
                current_chunk_sentences = current_chunk_sentences[-2:]
                current_length = sum(len(sent) for sent in current_chunk_sentences)
                
        if current_chunk_sentences:
            # Flush remaining sentences
            chunk_text = " ".join(current_chunk_sentences)
            c = ChunkMeta(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                filename=filename,
                title=title,
                author=author,
                page=page_num,
                text=chunk_text,
                tokens=len(chunk_text.split()),
                created_at=now_iso()
            )
            chunks.append(c)
            
    return chunks
