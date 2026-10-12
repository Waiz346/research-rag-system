"""Text chunker with metadata preservation and sliding window overlap."""
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from src.config import settings

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class DocumentChunker:
    """Splits long text documents into overlapping semantic chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(self, text: str, base_metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text into chunks by paragraphs or sliding window with metadata attached."""
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return []

        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_idx = 0

        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))
            
            # Try to snap to sentence end if within reasonable distance
            if end < len(cleaned_text):
                period_pos = cleaned_text.rfind(". ", start, end)
                if period_pos != -1 and (period_pos - start) > (self.chunk_size * 0.6):
                    end = period_pos + 1

            chunk_content = cleaned_text[start:end].strip()
            if chunk_content:
                meta = {
                    **base_metadata,
                    "chunk_index": chunk_idx,
                    "start_char": start,
                    "end_char": end,
                }
                # Stringify lists in metadata for vector DB compatibility
                for k, v in meta.items():
                    if isinstance(v, list):
                        meta[k] = ", ".join(str(item) for item in v)

                doc_id = base_metadata.get("doc_id", "doc")
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"

                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    metadata=meta
                ))
                chunk_idx += 1

            start += step
            if end == len(cleaned_text):
                break

        return chunks
