"""End-to-End RAG retrieval with Hybrid Search (Dense + BM25), HyDE expansion, and relevance threshold cutoff.

Inspired by RAGFlow's "Quality in, quality out" principles:
- Rejects low-confidence distractor chunks below a configurable relevance threshold.
- Employs Hybrid Retrieval (Dense vector semantic search + BM25 keyword matching).
- Preserves paper section hierarchy (Abstract, Methodology, Experiments, etc.) in citations.
"""
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
    citation_count: int = 0
    section: str = "General"
    dense_score: float = 0.0
    bm25_score: float = 0.0
    search_type: str = "hybrid"
    metadata: Dict[str, Any] = None

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
    search_mode: str = "hybrid"
    filtered_out_count: int = 0
    score_threshold: float = 0.35

class RAGPipeline:
    """Orchestrates Hybrid query retrieval, confidence gating, and LLM synthesis."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        generator: Optional[LLMGenerator] = None,
        min_relevance_threshold: float = 0.35
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.generator = generator or LLMGenerator()
        self.min_relevance_threshold = min_relevance_threshold

    def query(
        self,
        question: str,
        top_k: int = 4,
        use_hyde: bool = True,
        search_mode: str = "hybrid",
        score_threshold: Optional[float] = None,
        hybrid_alpha: float = 0.5,
        history: Optional[List[Dict[str, str]]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> RAGResponse:
        """Execute RAG query flow with Hybrid retrieval, conversation history, and threshold gating."""
        start_time = time.perf_counter()
        hyde_passage: Optional[str] = None
        effective_threshold = score_threshold if score_threshold is not None else self.min_relevance_threshold

        # 1. Query Expansion via HyDE (Hypothetical Document Embeddings)
        search_queries = [question]
        if use_hyde:
            try:
                hyde_passage = self.generator.generate_hyde(question)
                if hyde_passage and hyde_passage != question:
                    search_queries.append(hyde_passage)
            except Exception as e:
                logger.warning(f"HyDE expansion skipped due to error: {e}")

        # 2. Retrieve candidates: Ground primary scores directly in user's question
        seen_chunks: Dict[str, SearchResult] = {}
        fetch_k = max(top_k * 2, 8)

        # Primary retrieval for user's question (optionally filtered by doc_ids)
        direct_results = self.vector_store.search(
            query=question,
            top_k=fetch_k,
            mode=search_mode,
            alpha=hybrid_alpha,
            score_threshold=0.0,
            doc_ids=doc_ids
        )
        for r in direct_results:
            seen_chunks[r.chunk_id] = r

        # Secondary candidate discovery via HyDE
        if use_hyde and hyde_passage:
            hyde_results = self.vector_store.search(
                query=hyde_passage,
                top_k=fetch_k,
                mode=search_mode,
                alpha=hybrid_alpha,
                score_threshold=0.0,
                doc_ids=doc_ids
            )
            for hr in hyde_results:
                if hr.chunk_id in seen_chunks:
                    # Blend direct score and hyde score
                    d = seen_chunks[hr.chunk_id]
                    d.score = round(0.7 * d.score + 0.3 * hr.score, 4)
                else:
                    # Candidate only matched hypothetical passage; discount to avoid hallucinated match
                    hr.score = round(hr.score * 0.4, 4)
                    seen_chunks[hr.chunk_id] = hr

        all_candidates = sorted(seen_chunks.values(), key=lambda x: x.score, reverse=True)

        if not all_candidates:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                query=question,
                answer="No relevant research papers found in the index. Please ingest relevant papers first.",
                sources=[],
                latency_ms=round(latency_ms, 2),
                total_retrieved=0,
                confidence_score=0.0,
                is_low_confidence=True,
                warning_note="Vector index is empty or yielded zero matches.",
                search_mode=search_mode,
                filtered_out_count=0,
                score_threshold=effective_threshold
            )

        max_score = all_candidates[0].score if all_candidates else 0.0

        # 3. Quality Gating: Relevance Threshold Cutoff (RAGFlow Quality-in Quality-out)
        qualifying_results = [r for r in all_candidates if r.score >= effective_threshold]
        filtered_out_count = len(all_candidates) - len(qualifying_results)

        if not qualifying_results:
            latency_ms = (time.perf_counter() - start_time) * 1000
            top_candidate = all_candidates[0]
            top_title = top_candidate.metadata.get("title", "Unknown Document")
            top_sec = top_candidate.section

            warning_note = (
                f"Relevance Threshold Gated: Top retrieved chunk ('{top_title}', Section: {top_sec}) "
                f"scored {max_score:.2f}, which is below the required cutoff of {effective_threshold:.2f}."
            )
            logger.info(warning_note)

            rejection_message = (
                f"No retrieved paper passages met the required relevance threshold (cutoff: {effective_threshold:.2f}).\n\n"
                f"- **Top candidate found**: *{top_title}* ([{top_sec}])\n"
                f"- **Relevance score**: {max_score:.2f} / 1.00 (below threshold)\n\n"
                "The current indexed literature does not appear to contain verified sources directly addressing your question. "
                "To obtain an accurate answer without hallucinations, consider:\n"
                "1. **Ingesting papers** covering this specific research domain or topic.\n"
                "2. **Lowering the relevance cutoff** in the search settings if you wish to inspect weaker candidates.\n"
                "3. **Using BM25 or Hybrid search** with specific paper titles or author names."
            )

            return RAGResponse(
                query=question,
                answer=rejection_message,
                sources=[],
                latency_ms=round(latency_ms, 2),
                total_retrieved=0,
                confidence_score=max_score,
                is_low_confidence=True,
                warning_note=warning_note,
                hyde_passage=hyde_passage,
                search_mode=search_mode,
                filtered_out_count=filtered_out_count,
                score_threshold=effective_threshold
            )

        # 4. Context Assembly from Qualifying Chunks Only
        selected_results = qualifying_results[:top_k]
        context_blocks: List[str] = []
        citations: List[SourceCitation] = []

        for idx, item in enumerate(selected_results, start=1):
            title = item.metadata.get("title", "Untitled Document")
            paper_id = item.metadata.get("doc_id", item.chunk_id)
            c_count = int(item.metadata.get("citation_count", 0))
            sec = item.section or item.metadata.get("section", "General")

            snippet = item.text[:300] + ("..." if len(item.text) > 300 else "")
            citations.append(SourceCitation(
                index=idx,
                paper_id=str(paper_id),
                title=str(title),
                snippet=snippet,
                score=item.score,
                citation_count=c_count,
                section=sec,
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
                search_type=item.search_type,
                metadata=item.metadata
            ))

            cite_label = f", Citations: {c_count:,}" if c_count > 0 else ""
            score_label = f"Score: {item.score:.2f} (Dense: {item.dense_score:.2f}, BM25: {item.bm25_score:.2f})"
            context_blocks.append(
                f"[Source {idx}] Title: {title} | Section: {sec} (ID: {paper_id}{cite_label}, {score_label})\nContent: {item.text}"
            )

        context_str = "\n\n".join(context_blocks)

        # 5. Intent-aware & Conversational prompt construction
        history_str = ""
        if history:
            turns = []
            for h in history[-4:]:
                r_label = "User" if h.get("role") == "user" else "Assistant"
                turns.append(f"{r_label}: {h.get('content', '')}")
            if turns:
                history_str = "Prior Conversation Turns:\n" + "\n".join(turns) + "\n\n"

        prompt = (
            f"{history_str}"
            f"Context from research papers:\n{context_str}\n\n"
            f"Current User Question: {question}\n\n"
        )
        if max_score < 0.50:
            prompt += (
                "NOTE ON RELEVANCE: The retrieved passages have moderate relevance. "
                "Make sure you clearly state what aspects are supported by the text and avoid speculation.\n\n"
            )

        prompt += (
            "Synthesize a clear, accurate, conversational response based strictly on the provided context. "
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
            is_low_confidence=(max_score < 0.50),
            warning_note=None,
            hyde_passage=hyde_passage,
            search_mode=search_mode,
            filtered_out_count=filtered_out_count,
            score_threshold=effective_threshold
        )
