"""Pydantic schemas for request and response validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    query: str = Field(default="Retrieval Augmented Generation", description="Main topic or keyword to query ArXiv")
    seed_citation: Optional[str] = Field(default=None, description="Landmark paper citation (ArXiv ID or title, e.g. 2005.11401) to find papers citing it")
    narrow_keywords: Optional[str] = Field(default=None, description="Keywords to narrow down retrieved papers (comma-separated, e.g. hallucination, medical)")
    title_keyword: Optional[str] = Field(default=None, description="Keywords that must appear in the paper title (ti:)")
    abstract_keyword: Optional[str] = Field(default=None, description="Keywords that must appear in the paper abstract (abs:)")
    min_citations: int = Field(default=0, ge=0, description="Minimum citations required via Semantic Scholar")
    max_papers: int = Field(default=3, ge=1, le=10, description="Maximum number of papers to index")
    download_pdf: bool = Field(default=False, description="Whether to parse full PDF text or abstracts")
    section_aware: bool = Field(default=True, description="Enable academic paper section-aware chunking (RAGFlow style)")
    exclude_references: bool = Field(default=True, description="Exclude references section to prevent citation list noise")

class PaperSummaryInfo(BaseModel):
    id: str
    title: str
    citation_count: int = 0

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message author role: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

class DocumentSummarySchema(BaseModel):
    doc_id: str
    title: str
    source_type: str
    filename: str
    authors: str = "Unknown"
    citation_count: int = 0
    chunk_count: int = 0
    sections: List[str] = []
    created_at: str = ""
    char_count: int = 0

class DocumentDetailSchema(BaseModel):
    summary: DocumentSummarySchema
    full_text: str

class UploadResponse(BaseModel):
    status: str
    doc_id: str
    title: str
    chunks_indexed: int
    total_in_index: int
    sections: List[str]
    char_count: int

class IngestResponse(BaseModel):
    status: str
    papers_fetched: int
    chunks_indexed: int
    total_in_index: int
    paper_titles: List[str]
    papers: List[PaperSummaryInfo] = []

class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of retrieved chunks")
    use_hyde: bool = Field(default=True, description="Enable Hypothetical Document Embeddings for query expansion")
    search_mode: str = Field(default="hybrid", description="Retrieval mode: hybrid (Dense+BM25), dense, or sparse (BM25)")
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Minimum relevance threshold cutoff (RAGFlow quality gate)")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Dense vs BM25 weight (1.0 = pure dense, 0.0 = pure BM25)")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Prior conversation history turns for NotebookLM-style multi-turn chat")
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional paper IDs to filter conversation to specific sources")

class CitationSchema(BaseModel):
    index: int
    paper_id: str
    title: str
    snippet: str
    score: float
    citation_count: int = 0
    section: str = "General"
    dense_score: Optional[float] = 0.0
    bm25_score: Optional[float] = 0.0
    search_type: Optional[str] = "hybrid"
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
    search_mode: str = "hybrid"
    filtered_out_count: int = 0
    score_threshold: float = 0.35

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

class LLMConfigUpdate(BaseModel):
    provider: str = Field(..., description="Provider name: qwen, glm, gemini, openai, ollama, custom, or mock")
    model_name: Optional[str] = Field(default=None, description="Model identifier (e.g. qwen-plus, glm-4-flash, gpt-4o-mini)")
    api_key: Optional[str] = Field(default=None, description="API Key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL endpoint for OpenAI-compatible providers")

class LLMConfigResponse(BaseModel):
    provider: str
    model_name: str
    base_url: str
    has_api_key: bool
    masked_api_key: str

class LLMTestResponse(BaseModel):
    success: bool
    provider: str
    model_name: str
    latency_ms: float
    message: str
    sample_output: Optional[str] = None
    error: Optional[str] = None
