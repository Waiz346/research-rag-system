"""End-to-End RAG retrieval, HyDE query expansion, and synthesis pipeline."""
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from src.indexing.vector_store import VectorStoreManager, SearchResult
from src.rag.generator import LLMGenerator

logger = logging.getLogger(__name__)

@dataclass
class SourceCitation:
    index: int
    paper_id: str
    title: str
    snippet: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: List[SourceCitation]
    latency_ms: float
    total_retrieved: int
    confidence_score: float = 0.0
    is_low_confidence: bool = False
    warning_note: Optional[str] = None
    hyde_passage: Optional[str] = None

class RAGPipeline:
    """Orchestrates query retrieval, HyDE expansion, confidence gating, and LLM synthesis."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        generator: Optional[LLMGenerator] = None,
        min_relevance_threshold: float = 0.40
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.generator = generator or LLMGenerator()
        self.min_relevance_threshold = min_relevance_threshold

    def query(
        self,
        question: str,
        top_k: int = 4,
        use_hyde: bool = True
    ) -> RAGResponse:
        """Execute full RAG query flow with HyDE expansion and confidence gating."""
        start_time = time.perf_counter()
        hyde_passage: Optional[str] = None

        # 1. Query Expansion via HyDE (Hypothetical Document Embeddings)
        search_queries = [question]
        if use_hyde:
            try:
                hyde_passage = self.generator.generate_hyde(question)
                if hyde_passage and hyde_passage != question:
                    search_queries.append(hyde_passage)
            except Exception as e:
                logger.warning(f"HyDE expansion skipped due to error: {e}")

        # 2. Retrieve candidates across queries and deduplicate
        seen_chunks: Dict[str, SearchResult] = {}
        for q in search_queries:
            results = self.vector_store.similarity_search(query=q, top_k=top_k)
            for r in results:
                if r.chunk_id not in seen_chunks or r.score > seen_chunks[r.chunk_id].score:
                    seen_chunks[r.chunk_id] = r

        # Sort deduplicated results by similarity score descending
        sorted_results = sorted(seen_chunks.values(), key=lambda x: x.score, reverse=True)[:top_k]

        if not sorted_results:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                query=question,
                answer="No relevant research papers found in the index. Please ingest relevant papers first.",
                sources=[],
                latency_ms=round(latency_ms, 2),
                total_retrieved=0,
                confidence_score=0.0,
                is_low_confidence=True,
                warning_note="Vector index is empty or yielded zero matches."
            )

        # 3. Confidence Gating & Thresholding
        max_score = sorted_results[0].score if sorted_results else 0.0
        avg_score = round(sum(r.score for r in sorted_results) / len(sorted_results), 4)
        is_low_confidence = max_score < self.min_relevance_threshold
        warning_note = None

        if is_low_confidence:
            warning_note = (
                f"Low retrieval confidence (Top match score: {max_score:.2f} < {self.min_relevance_threshold:.2f}). "
                "The current index appears to lack foundational papers on this general query."
            )
            logger.warning(warning_note)

        # 4. Build context blocks and source citations
        context_blocks: List[str] = []
        citations: List[SourceCitation] = []

        for idx, item in enumerate(sorted_results, start=1):
            title = item.metadata.get("title", "Untitled Document")
            paper_id = item.metadata.get("doc_id", item.chunk_id)

            snippet = item.text[:300] + ("..." if len(item.text) > 300 else "")
            citations.append(SourceCitation(
                index=idx,
                paper_id=str(paper_id),
                title=str(title),
                snippet=snippet,
                score=item.score,
                metadata=item.metadata
            ))

            context_blocks.append(
                f"[Source {idx}] Title: {title} (ID: {paper_id}, Similarity: {item.score:.2f})\nContent: {item.text}"
            )

        context_str = "\n\n".join(context_blocks)
        
        # 5. Intent-aware prompt construction
        prompt = (
            f"Context from research papers:\n{context_str}\n\n"
            f"User Question: {question}\n\n"
        )
        if is_low_confidence:
            prompt += (
                "NOTE ON RELEVANCE: The retrieved passages have borderline relevance. "
                "If the context only mentions specific sub-variants (e.g. for image generation) "
                "and lacks general definitions, make this limitation explicitly clear to the user.\n\n"
            )

        prompt += (
            "Synthesize a clear, accurate response based strictly on the provided context. "
            "Cite all claims using [Source X] format."
        )

        answer = self.generator.generate(prompt)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return RAGResponse(
            query=question,
            answer=answer,
            sources=citations,
            latency_ms=round(latency_ms, 2),
            total_retrieved=len(citations),
            confidence_score=max_score,
            is_low_confidence=is_low_confidence,
            warning_note=warning_note,
            hyde_passage=hyde_passage
        )
