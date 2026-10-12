"""ChromaDB vector store manager for indexing and semantic retrieval."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import settings
from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    score: float  # Normalized similarity score (1.0 - normalized distance)

class VectorStoreManager:
    """Manages persistent embedding indexing and similarity search via ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        logger.info(f"Connecting to ChromaDB at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "ArXiv Research Papers Index"}
        )

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Upsert text chunks with their metadata into the vector collection."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Indexed {len(chunks)} chunks into collection '{self.collection_name}'.")
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Perform semantic retrieval for a user question."""
        total_count = self.collection.count()
        if total_count == 0:
            logger.warning("Vector store is empty! No documents have been indexed yet.")
            return []

        limit = min(top_k, total_count)
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )

        search_results: List[SearchResult] = []
        if results and results["documents"]:
            docs = results["documents"][0]
            ids = results["ids"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
                # Cosine distance ranges roughly from 0 to 2; normalize score between 0 and 1
                sim_score = max(0.0, 1.0 - (dist / 2.0))
                search_results.append(SearchResult(
                    chunk_id=doc_id,
                    text=text,
                    metadata=meta,
                    distance=dist,
                    score=round(sim_score, 4)
                ))

        return search_results

    def count(self) -> int:
        """Return total indexed items in collection."""
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "ArXiv Research Papers Index"}
        )
        logger.info(f"Reset collection '{self.collection_name}'.")
