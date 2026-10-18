"""End-to-End RAG retrieval with Hybrid Search (Dense + BM25), HyDE expansion, and relevance threshold cutoff.

Inspired by RAGFlow's "Quality in, quality out" principles:
- Rejects low-confidence distractor chunks below a configurable relevance threshold.
- Employs Hybrid Retrieval (Dense vector semantic search + BM25 keyword matching).
- Preserves paper section hierarchy (Abstract, Methodology, Experiments, etc.) in citations.
"""
import time
import logging
from dataclasses import dataclass, field
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
    rewritten_query: Optional[str] = None
    query_intent: str = "conceptual_research"
    follow_up_questions: List[str] = field(default_factory=list)

class RAGPipeline:
    """Orchestrates Hybrid query retrieval, conversational rewriting, diversity reranking, and LLM synthesis."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        generator: Optional[LLMGenerator] = None,
        min_relevance_threshold: float = 0.35
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.generator = generator or LLMGenerator()
        self.min_relevance_threshold = min_relevance_threshold

    @staticmethod
    def _select_diverse_top_k(
        qualifying_results: List[SearchResult],
        top_k: int,
        prefer_diversity: bool = False
    ) -> List[SearchResult]:
        """Select top_k chunks while ensuring multi-document diversity when multiple sources qualify."""
        if len(qualifying_results) <= top_k:
            return qualifying_results

        top_score = qualifying_results[0].score
        # Group candidates by document ID
        by_doc: Dict[str, List[SearchResult]] = {}
        for r in qualifying_results:
            doc_id = str((r.metadata or {}).get("doc_id") or r.chunk_id)
            by_doc.setdefault(doc_id, []).append(r)

        if len(by_doc) <= 1:
            return qualifying_results[:top_k]

        # If broad synthesis or multiple competitive documents exist, round-robin the top chunk from each competitive doc first
        min_competitive_score = top_score * (0.55 if prefer_diversity else 0.72)
        selected: List[SearchResult] = []
        seen_ids = set()

        for doc_id, doc_chunks in by_doc.items():
            best_chunk = doc_chunks[0]
            if best_chunk.score >= min_competitive_score and len(selected) < top_k:
                selected.append(best_chunk)
                seen_ids.add(best_chunk.chunk_id)

        # Fill remaining slots by highest overall score
        for r in qualifying_results:
            if len(selected) >= top_k:
                break
            if r.chunk_id not in seen_ids:
                selected.append(r)
                seen_ids.add(r.chunk_id)

        selected.sort(key=lambda x: x.score, reverse=True)
        return selected

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
        """Execute RAG query flow with Conversational Rewriting, Hybrid retrieval, Diversity Reranking, and threshold gating."""
        start_time = time.perf_counter()
        hyde_passage: Optional[str] = None
        effective_threshold = score_threshold if score_threshold is not None else self.min_relevance_threshold

        # 0. Conversational Query Rewriting (Resolves "it", "that paper", "the lecture" from chat history)
        effective_query = self.generator.rewrite_conversational_query(question, history)
        rewritten_query = effective_query if effective_query != question else None
        query_intent = self.generator.classify_query_intent(effective_query)

        # 1. Intent-Adaptive Query Expansion via HyDE (Skipped for fast coursework/deadline factual lookups)
        if use_hyde and query_intent != "coursework_factual":
            try:
                hyde_passage = self.generator.generate_hyde(effective_query)
                if hyde_passage == effective_query:
                    hyde_passage = None
            except Exception as e:
                logger.warning(f"HyDE expansion skipped due to error: {e}")

        # 2. Retrieve candidates: Ground primary scores in effective_query (and raw question if rewritten)
        seen_chunks: Dict[str, SearchResult] = {}
        fetch_k = max(top_k * 3, 10)

        # Primary retrieval for effective search query (optionally filtered by doc_ids)
        direct_results = self.vector_store.search(
            query=effective_query,
            top_k=fetch_k,
            mode=search_mode,
            alpha=hybrid_alpha,
            score_threshold=0.0,
            doc_ids=doc_ids
        )
        for r in direct_results:
            seen_chunks[r.chunk_id] = r

        # If conversational rewrite modified the query, also merge direct hits for raw question
        if rewritten_query:
            raw_results = self.vector_store.search(
                query=question,
                top_k=fetch_k,
                mode=search_mode,
                alpha=hybrid_alpha,
                score_threshold=0.0,
                doc_ids=doc_ids
            )
            for rr in raw_results:
                if rr.chunk_id in seen_chunks:
                    seen_chunks[rr.chunk_id].score = max(seen_chunks[rr.chunk_id].score, rr.score)
                else:
                    seen_chunks[rr.chunk_id] = rr

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
                score_threshold=effective_threshold,
                rewritten_query=rewritten_query,
                query_intent=query_intent
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
                score_threshold=effective_threshold,
                rewritten_query=rewritten_query,
                query_intent=query_intent
            )

        # 4. NotebookLM-Style Hierarchical Context Assembly:
        #    Tier 1: Global Source Map (so the LLM knows all active sources in the notebook)
        #    Tier 2: Diverse Top-K Passages with Neighbor Window Expansion (stitching chunk i-1 & i+1)
        prefer_diversity = (query_intent == "broad_synthesis") or (doc_ids is not None and len(doc_ids) > 1)
        selected_results = self._select_diverse_top_k(
            qualifying_results,
            top_k=top_k,
            prefer_diversity=prefer_diversity
        )
        context_blocks: List[str] = []
        citations: List[SourceCitation] = []

        for idx, item in enumerate(selected_results, start=1):
            meta = item.metadata or {}
            title = meta.get("title", "Untitled Document")
            paper_id = meta.get("doc_id", item.chunk_id)
            c_count = int(meta.get("citation_count", 0))
            sec = item.section or meta.get("section", "General")
            loc_label = meta.get("location_label", "")
            sec_with_loc = f"{sec} • {loc_label}" if loc_label and loc_label.lower() not in sec.lower() else sec

            # Expand passage with neighboring chunk context (chunk i-1 / i+1) so arguments/tables aren't cut off
            expanded_text = self.vector_store.expand_chunk_context(item.chunk_id) or item.text

            snippet = item.text[:420] + ("..." if len(item.text) > 420 else "")
            citations.append(SourceCitation(
                index=idx,
                paper_id=str(paper_id),
                title=str(title),
                snippet=snippet,
                score=item.score,
                citation_count=c_count,
                section=sec_with_loc,
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
                search_type=item.search_type,
                metadata=meta
            ))

            cite_label = f", Citations: {c_count:,}" if c_count > 0 else ""
            score_label = f"Score: {item.score:.2f} (Dense: {item.dense_score:.2f}, BM25: {item.bm25_score:.2f})"
            context_blocks.append(
                f"[Source {idx}] Title: {title} | Section: {sec_with_loc} (ID: {paper_id}{cite_label}, {score_label})\nContent: {expanded_text}"
            )

        context_str = "\n\n".join(context_blocks)

        # Build Tier-1 Global Notebook Source Map
        source_map_str = ""
        active_overviews = self.vector_store.get_active_sources_overview(doc_ids=doc_ids, max_sources=8)
        if active_overviews:
            map_lines = [
                f"- {ov['title']} (ID: {ov['doc_id']}, Authors: {ov['authors']}): {ov['overview']}"
                for ov in active_overviews
            ]
            source_map_str = "Notebook Active Sources Map (High-Level Overview):\n" + "\n".join(map_lines) + "\n\n"

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
            f"{source_map_str}"
            f"Context from indexed documents (Expanded Passage Windows):\n{context_str}\n\n"
            f"Current User Question: {question}\n\n"
        )
        if rewritten_query:
            prompt += f"(Contextualized Search Intent: {rewritten_query})\n\n"

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

        # 6. Build 3 NotebookLM-style grounded follow-up questions
        follow_ups = self._build_follow_up_questions(question, citations)
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
            score_threshold=effective_threshold,
            rewritten_query=rewritten_query,
            query_intent=query_intent,
            follow_up_questions=follow_ups
        )

    @staticmethod
    def _build_follow_up_questions(question: str, citations: List[SourceCitation]) -> List[str]:
        """Generate 3 source-grounded follow-up questions tailored to the retrieved sources."""
        if not citations:
            return []
        unique_titles: List[str] = []
        sections_seen: List[str] = []
        for c in citations:
            short_t = c.title[:55] + ("..." if len(c.title) > 55 else "")
            if short_t not in unique_titles:
                unique_titles.append(short_t)
            sec_clean = (c.section or "").split("•")[0].strip()
            if sec_clean and sec_clean != "General" and sec_clean not in sections_seen:
                sections_seen.append(sec_clean)

        follow_ups: List[str] = []
        if len(unique_titles) >= 2:
            follow_ups.append(f"How do \"{unique_titles[0]}\" and \"{unique_titles[1]}\" compare in their methodology and findings?")
        else:
            follow_ups.append(f"What are the main limitations or failure modes discussed in \"{unique_titles[0]}\"?")

        if sections_seen:
            follow_ups.append(f"Can you break down the technical details from the {sections_seen[0]} section?")
        else:
            follow_ups.append(f"What empirical benchmarks or evaluation metrics support these claims?")

        follow_ups.append(f"What practical takeaways or open research questions emerge from these sources regarding: {question[:45]}?")
        return follow_ups[:3]
