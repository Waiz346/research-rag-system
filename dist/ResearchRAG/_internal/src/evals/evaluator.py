"""Evaluation metrics for RAG systems: Semantic LLM-as-a-Judge with heuristic fallback."""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from src.rag.pipeline import RAGResponse
from src.rag.generator import LLMGenerator

logger = logging.getLogger(__name__)

@dataclass
class TestSample:
    query: str
    expected_answer: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)

@dataclass
class EvaluationMetric:
    score: float  # Range: 0.0 to 1.0
    rationale: str

@dataclass
class EvaluationResult:
    query: str
    context_relevance: EvaluationMetric
    faithfulness: EvaluationMetric
    answer_relevance: EvaluationMetric
    composite_score: float
    evaluation_mode: str = "heuristic"  # "llm_judge" or "heuristic"

class RAGEvaluator:
    """Evaluates RAG pipeline outputs using LLM-as-a-Judge semantic grading with heuristic fallback."""

    def __init__(self, generator: Optional[LLMGenerator] = None):
        self.generator = generator or LLMGenerator()

    @staticmethod
    def _extract_tokens(text: str) -> set:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "this", "that", "with", "from", "for", "are", "was",
            "were", "which", "what", "how", "why", "when", "where", "about"
        }
        return {t for t in tokens if t not in stopwords}

    def evaluate_response(
        self,
        response: RAGResponse,
        sample: Optional[TestSample] = None,
        prefer_llm_judge: bool = True
    ) -> EvaluationResult:
        """Run all evaluation metrics for a single RAG response using LLM judge or heuristic."""
        context_text = "\n".join([f"- {s.title}: {s.snippet}" for s in response.sources])
        query = response.query
        answer = response.answer

        # Try LLM-as-a-Judge first for semantic accuracy
        if prefer_llm_judge and self.generator.provider != "mock":
            try:
                judged = self.generator.judge_evaluation(
                    query=query,
                    answer=answer,
                    context=context_text
                )
                if judged:
                    c_score = max(0.0, min(1.0, float(judged["context_relevance"]["score"])))
                    f_score = max(0.0, min(1.0, float(judged["faithfulness"]["score"])))
                    a_score = max(0.0, min(1.0, float(judged["answer_relevance"]["score"])))

                    context_rel = EvaluationMetric(
                        score=round(c_score, 3),
                        rationale=f"[LLM Judge] {judged['context_relevance']['rationale']}"
                    )
                    faithfulness = EvaluationMetric(
                        score=round(f_score, 3),
                        rationale=f"[LLM Judge] {judged['faithfulness']['rationale']}"
                    )
                    answer_rel = EvaluationMetric(
                        score=round(a_score, 3),
                        rationale=f"[LLM Judge] {judged['answer_relevance']['rationale']}"
                    )

                    composite = round((0.35 * c_score) + (0.35 * f_score) + (0.30 * a_score), 3)

                    return EvaluationResult(
                        query=query,
                        context_relevance=context_rel,
                        faithfulness=faithfulness,
                        answer_relevance=answer_rel,
                        composite_score=composite,
                        evaluation_mode="llm_judge"
                    )
            except Exception as e:
                logger.warning(f"LLM-as-a-Judge failed, falling back to heuristic: {e}")

        # Fallback: Heuristic token overlap
        context_rel = self._evaluate_context_relevance(query, context_text)
        faithfulness = self._evaluate_faithfulness(answer, context_text)
        answer_rel = self._evaluate_answer_relevance(query, answer, sample)

        composite = round(
            (0.35 * context_rel.score) + (0.35 * faithfulness.score) + (0.30 * answer_rel.score),
            3
        )

        return EvaluationResult(
            query=query,
            context_relevance=context_rel,
            faithfulness=faithfulness,
            answer_relevance=answer_rel,
            composite_score=composite,
            evaluation_mode="heuristic"
        )

    def _evaluate_context_relevance(self, query: str, context_text: str) -> EvaluationMetric:
        """Heuristic measuring concept overlap between query and context."""
        if not context_text.strip():
            return EvaluationMetric(score=0.0, rationale="No context retrieved.")

        q_tokens = self._extract_tokens(query)
        if not q_tokens:
            return EvaluationMetric(score=1.0, rationale="Query contains no evaluable keywords.")

        ctx_tokens = self._extract_tokens(context_text)
        matched = q_tokens.intersection(ctx_tokens)
        ratio = len(matched) / len(q_tokens)
        score = round(min(1.0, max(0.0, ratio)), 3)

        return EvaluationMetric(
            score=score,
            rationale=f"Retrieved context matched {len(matched)} of {len(q_tokens)} key query concepts."
        )

    def _evaluate_faithfulness(self, answer: str, context_text: str) -> EvaluationMetric:
        """Heuristic checking if content words in answer are supported by sources."""
        if not answer.strip():
            return EvaluationMetric(score=0.0, rationale="Answer is empty.")
        if not context_text.strip():
            return EvaluationMetric(score=0.0, rationale="No context available to support answer.")

        ans_tokens = self._extract_tokens(answer)
        ctx_tokens = self._extract_tokens(context_text)

        if not ans_tokens:
            return EvaluationMetric(score=0.5, rationale="Answer contains minimal terms.")

        overlap = ans_tokens.intersection(ctx_tokens)
        grounded_ratio = len(overlap) / len(ans_tokens)
        score = round(min(1.0, max(0.0, grounded_ratio)), 3)

        return EvaluationMetric(
            score=score,
            rationale=f"{int(score * 100)}% of content words in answer appear in source text."
        )

    def _evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
        sample: Optional[TestSample] = None
    ) -> EvaluationMetric:
        """Heuristic checking alignment between question and answer."""
        q_tokens = self._extract_tokens(query)
        ans_tokens = self._extract_tokens(answer)

        if not ans_tokens:
            return EvaluationMetric(score=0.0, rationale="Empty answer.")

        overlap = q_tokens.intersection(ans_tokens)
        base_score = len(overlap) / max(1, len(q_tokens))

        if sample and sample.expected_keywords:
            expected_set = {k.lower() for k in sample.expected_keywords}
            expected_matches = expected_set.intersection(ans_tokens)
            keyword_score = len(expected_matches) / len(expected_set)
            final_score = round(0.5 * base_score + 0.5 * keyword_score, 3)
            return EvaluationMetric(
                score=min(1.0, final_score),
                rationale=f"Answer contains expected concepts ({len(expected_matches)}/{len(expected_set)})."
            )

        score = round(min(1.0, max(0.2, base_score)), 3)
        return EvaluationMetric(
            score=score,
            rationale=f"Answer shares {len(overlap)} concepts with user question."
        )
