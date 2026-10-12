"""Pydantic schemas for request and response validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    query: str = Field(..., example="agentic workflows large language models", description="Search topic to query ArXiv")
    max_papers: int = Field(default=3, ge=1, le=10, description="Number of papers to fetch")
    download_pdf: bool = Field(default=False, description="Whether to parse full PDF or abstracts")

class IngestResponse(BaseModel):
    status: str
    papers_fetched: int
    chunks_indexed: int
    total_in_index: int
    paper_titles: List[str]

class QueryRequest(BaseModel):
    question: str = Field(..., example="How do multi-agent systems collaborate?", description="Question to answer")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of retrieved chunks")
    use_hyde: bool = Field(default=True, description="Enable Hypothetical Document Embeddings for query expansion")

class CitationSchema(BaseModel):
    index: int
    paper_id: str
    title: str
    snippet: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[CitationSchema]
    latency_ms: float
    total_retrieved: int
    confidence_score: float = 0.0
    is_low_confidence: bool = False
    warning_note: Optional[str] = None
    hyde_passage: Optional[str] = None

class MetricDetail(BaseModel):
    score: float
    rationale: str

class EvaluationReport(BaseModel):
    query: str
    answer: str
    context_relevance: MetricDetail
    faithfulness: MetricDetail
    answer_relevance: MetricDetail
    composite_score: float
    evaluation_mode: str = "llm_judge"

class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_provider: str
    indexed_documents_count: int
