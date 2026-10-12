"""ChromaDB vector store & BM25 hybrid search engine with Reciprocal Rank Fusion (RRF).

Combines:
1. Dense Semantic Retrieval via ChromaDB vector embeddings.
2. Sparse Keyword Retrieval via BM25Okapi for exact terminology and identifiers.
3. Reciprocal Rank Fusion (RRF) and dynamic relevance threshold filtering.
"""
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from src.config import settings
from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
    'having', 'he', 'her', 'here', 'hers', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is',
    'it', 'its', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'our', 'ours', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'then', 'there', 'these', 'they', 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'with', 'you', 'your', 'yours'
}

@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float = 0.0
    score: float = 0.0              # Primary composite score (0.0 to 1.0)
    dense_score: float = 0.0        # Vector cosine similarity (0.0 to 1.0)
    bm25_score: float = 0.0         # Normalized BM25 score (0.0 to 1.0)
    rrf_score: float = 0.0          # Reciprocal Rank Fusion score (0.0 to 1.0)
    search_type: str = "hybrid"     # "hybrid", "dense", or "sparse"
    section: str = "General"        # Section label (e.g. "Abstract", "Methodology")

class VectorStoreManager:
    """Manages persistent embedding indexing and Hybrid (Dense + BM25) retrieval."""

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

        # In-memory BM25 index structures
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_doc_ids: List[str] = []
        self.bm25_docs: Dict[str, Dict[str, Any]] = {}

        # Hydrate BM25 from existing persistent ChromaDB documents
        self._build_bm25_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple, fast lowercase alphanumeric word tokenizer."""
        return re.findall(r"\w+", text.lower())

    def _build_bm25_index(self):
        """Build or re-build BM25 index from all documents currently in the collection."""
        try:
            stored = self.collection.get()
            if not stored or not stored.get("ids"):
                self.bm25 = None
                self.bm25_doc_ids = []
                self.bm25_docs = {}
                return

            ids = stored["ids"]
            documents = stored.get("documents", [])
            metadatas = stored.get("metadatas", []) or [{}] * len(ids)

            tokenized_corpus: List[List[str]] = []
            valid_ids: List[str] = []
            doc_map: Dict[str, Dict[str, Any]] = {}

            for doc_id, doc_text, meta in zip(ids, documents, metadatas):
                tokens = self._tokenize(doc_text)
                tokenized_corpus.append(tokens)
                valid_ids.append(doc_id)
                doc_map[doc_id] = {
                    "text": doc_text,
                    "metadata": meta or {},
                    "section": (meta or {}).get("section", "General")
                }

            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.bm25_doc_ids = valid_ids
                self.bm25_docs = doc_map
                logger.info(f"BM25 index synchronized with {len(valid_ids)} chunks.")
            else:
                self.bm25 = None
                self.bm25_doc_ids = []
                self.bm25_docs = {}
        except Exception as e:
            logger.warning(f"Could not build BM25 index from ChromaDB: {e}")
            self.bm25 = None

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Upsert text chunks with their metadata into ChromaDB and update BM25 index."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # 1. Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        # 2. Update in-memory BM25 index
        self._build_bm25_index()

        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB & BM25 collection '{self.collection_name}'.")
        return len(chunks)

    def dense_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform dense embedding retrieval via ChromaDB."""
        total_count = self.collection.count()
        if total_count == 0:
            return []

        limit = min(max(top_k, 1), total_count)
        where_filter = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter
        )

        search_results: List[SearchResult] = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            ids = results["ids"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
                # Cosine distance ranges [0, 2]; normalize score to [0, 1]
                sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                if sim_score < score_threshold:
                    continue

                sec = (meta or {}).get("section", "General")
                search_results.append(SearchResult(
                    chunk_id=doc_id,
                    text=text,
                    metadata=meta,
                    distance=dist,
                    score=round(sim_score, 4),
                    dense_score=round(sim_score, 4),
                    bm25_score=0.0,
                    rrf_score=0.0,
                    search_type="dense",
                    section=sec
                ))

        return search_results

    def bm25_search(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform sparse keyword retrieval via BM25Okapi."""
        if self.bm25 is None or not self.bm25_doc_ids:
            return []

        all_tokens = self._tokenize(query)
        # Filter stopwords to prioritize discriminative domain concepts
        filtered = [t for t in all_tokens if t not in ENGLISH_STOPWORDS]
        query_tokens = filtered if filtered else all_tokens
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        max_score = float(max(scores)) if len(scores) > 0 else 0.0
        if max_score <= 0.0:
            return []

        # Soft saturation curve x / (x + 5.0) prevents tiny 1-stopword matches from blowing up to 1.0
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[SearchResult] = []

        for idx in ranked_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0.0:
                continue

            doc_id = self.bm25_doc_ids[idx]
            doc_data = self.bm25_docs.get(doc_id, {})
            meta = doc_data.get("metadata", {})
            if doc_ids and meta.get("doc_id") not in doc_ids:
                continue

            norm_score = raw_score / (raw_score + 5.0)
            if norm_score < score_threshold:
                continue

            text = doc_data.get("text", "")
            sec = doc_data.get("section", "General")

            results.append(SearchResult(
                chunk_id=doc_id,
                text=text,
                metadata=meta,
                distance=0.0,
                score=round(norm_score, 4),
                dense_score=0.0,
                bm25_score=round(norm_score, 4),
                rrf_score=0.0,
                search_type="sparse",
                section=sec
            ))

            if len(results) >= top_k:
                break

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        alpha: float = 0.5,
        score_threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform Hybrid Search combining Dense & BM25 with Reciprocal Rank Fusion (RRF).

        alpha (0.0 to 1.0): Weight for dense vector search. (1 - alpha) is weight for BM25.
        """
        fetch_k = max(top_k * 3, 10)
        dense_candidates = self.dense_search(query, top_k=fetch_k, score_threshold=0.0, doc_ids=doc_ids)
        bm25_candidates = self.bm25_search(query, top_k=fetch_k, score_threshold=0.0, doc_ids=doc_ids)

        if not dense_candidates and not bm25_candidates:
            return []
        if not dense_candidates:
            return [c for c in bm25_candidates if c.score >= score_threshold][:top_k]
        if not bm25_candidates:
            return [c for c in dense_candidates if c.score >= score_threshold][:top_k]

        # Build rank dictionaries (1-based index)
        dense_ranks: Dict[str, int] = {c.chunk_id: idx + 1 for idx, c in enumerate(dense_candidates)}
        dense_scores: Dict[str, float] = {c.chunk_id: c.dense_score for c in dense_candidates}

        bm25_ranks: Dict[str, int] = {c.chunk_id: idx + 1 for idx, c in enumerate(bm25_candidates)}
        bm25_scores: Dict[str, float] = {c.chunk_id: c.bm25_score for c in bm25_candidates}

        # Union of candidate IDs
        all_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))

        # Reciprocal Rank Fusion constant
        k_rrf = 60.0
        w_dense = alpha
        w_bm25 = 1.0 - alpha
        max_possible_rrf = (w_dense / (k_rrf + 1.0)) + (w_bm25 / (k_rrf + 1.0))

        fused_results: List[SearchResult] = []

        for cid in all_ids:
            r_dense = dense_ranks.get(cid, 9999)
            r_bm25 = bm25_ranks.get(cid, 9999)

            d_score = dense_scores.get(cid, 0.0)
            b_score = bm25_scores.get(cid, 0.0)

            # RRF calculation
            rrf_raw = (w_dense / (k_rrf + r_dense)) + (w_bm25 / (k_rrf + r_bm25))
            rrf_norm = min(1.0, rrf_raw / max_possible_rrf)

            # Fused normalized score combining similarity & keyword alignment
            fused_score = (w_dense * d_score) + (w_bm25 * b_score)
            composite_score = round((0.6 * fused_score) + (0.4 * rrf_norm), 4)

            if composite_score < score_threshold:
                continue

            # Retrieve text and metadata from either candidate pool
            if cid in self.bm25_docs:
                text = self.bm25_docs[cid]["text"]
                meta = self.bm25_docs[cid]["metadata"]
                sec = self.bm25_docs[cid]["section"]
            else:
                dense_match = next((c for c in dense_candidates if c.chunk_id == cid), None)
                text = dense_match.text if dense_match else ""
                meta = dense_match.metadata if dense_match else {}
                sec = (meta or {}).get("section", "General")

            fused_results.append(SearchResult(
                chunk_id=cid,
                text=text,
                metadata=meta,
                distance=0.0,
                score=composite_score,
                dense_score=round(d_score, 4),
                bm25_score=round(b_score, 4),
                rrf_score=round(rrf_norm, 4),
                search_type="hybrid",
                section=sec
            ))

        # Sort by primary composite score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 4,
        mode: str = "hybrid",
        alpha: float = 0.5,
        score_threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Unified search method routing to hybrid, dense, or sparse retrieval."""
        clean_mode = (mode or "hybrid").strip().lower()
        if clean_mode in ["sparse", "bm25"]:
            return self.bm25_search(query, top_k=top_k, score_threshold=score_threshold, doc_ids=doc_ids)
        elif clean_mode in ["dense", "vector"]:
            return self.dense_search(query, top_k=top_k, score_threshold=score_threshold, doc_ids=doc_ids)
        else:
            return self.hybrid_search(query, top_k=top_k, alpha=alpha, score_threshold=score_threshold, doc_ids=doc_ids)

    def similarity_search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Backward-compatible dense similarity search alias."""
        return self.dense_search(query, top_k=top_k, score_threshold=0.0)

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """Delete all indexed chunks for a document ID from ChromaDB and BM25."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
            self._build_bm25_index()
            logger.info(f"Deleted chunks for doc_id='{doc_id}' from vector store.")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for doc_id='{doc_id}': {e}")
            return False

    def count(self) -> int:
        """Return total indexed items in collection."""
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection and clear BM25 index."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "ArXiv Research Papers Index"}
        )
        self.bm25 = None
        self.bm25_doc_ids = []
        self.bm25_docs = {}
        logger.info(f"Reset collection '{self.collection_name}' and BM25 index.")
