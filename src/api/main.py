"""FastAPI entry point for the Research RAG Intelligence system."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.ingestion.arxiv_collector import ArxivCollector
from src.ingestion.chunker import DocumentChunker
from src.indexing.vector_store import VectorStoreManager
from src.rag.pipeline import RAGPipeline
from src.rag.generator import LLMGenerator
from src.evals.evaluator import RAGEvaluator
from src.api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, CitationSchema,
    EvaluationReport, MetricDetail,
    HealthResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# State containers
vector_store = None
pipeline = None
evaluator = None

def get_vector_store() -> VectorStoreManager:
    global vector_store
    if vector_store is None:
        vector_store = VectorStoreManager()
    return vector_store

def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        generator = LLMGenerator()
        pipeline = RAGPipeline(vector_store=get_vector_store(), generator=generator)
    return pipeline

def get_evaluator() -> RAGEvaluator:
    global evaluator
    if evaluator is None:
        generator = LLMGenerator()
        evaluator = RAGEvaluator(generator=generator)
    return evaluator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook to initialize pipeline components on startup."""
    logger.info("Initializing Vector Store and RAG pipeline...")
    get_vector_store()
    get_pipeline()
    get_evaluator()
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="Research RAG Intelligence System",
    description="End-to-End ArXiv Paper Ingestion, Semantic Retrieval, and Automated RAG Evaluation API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", summary="Root endpoint")
def root():
    return {
        "message": "Research RAG Intelligence System is online.",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, summary="Check service health")
def health():
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        llm_provider=settings.LLM_PROVIDER,
        indexed_documents_count=get_vector_store().count()
    )

@app.post("/ingest", response_model=IngestResponse, summary="Ingest papers from ArXiv")
def ingest_papers(req: IngestRequest):
    """Fetch ArXiv research papers, split into overlapping chunks, and index into ChromaDB."""
    try:
        collector = ArxivCollector()
        papers = collector.search_papers(
            query=req.query,
            max_results=req.max_papers,
            download_pdf=req.download_pdf
        )

        if not papers:
            raise HTTPException(status_code=404, detail="No papers found matching the query.")

        chunker = DocumentChunker()
        all_chunks = []
        titles = []

        for paper in papers:
            titles.append(paper.title)
            base_meta = {
                "doc_id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "pdf_url": paper.pdf_url
            }
            chunks = chunker.split_text(paper.text_content, base_metadata=base_meta)
            all_chunks.extend(chunks)

        store = get_vector_store()
        indexed_count = store.add_chunks(all_chunks)

        return IngestResponse(
            status="success",
            papers_fetched=len(papers),
            chunks_indexed=indexed_count,
            total_in_index=store.count(),
            paper_titles=titles
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse, summary="Query RAG pipeline")
def query_rag(req: QueryRequest):
    """Perform similarity search and synthesize answer with source citations."""
    try:
        pipeline_inst = get_pipeline()
        res = pipeline_inst.query(question=req.question, top_k=req.top_k, use_hyde=req.use_hyde)
        
        citations = [
            CitationSchema(
                index=s.index,
                paper_id=s.paper_id,
                title=s.title,
                snippet=s.snippet,
                score=s.score,
                metadata=s.metadata
            )
            for s in res.sources
        ]

        return QueryResponse(
            query=res.query,
            answer=res.answer,
            sources=citations,
            latency_ms=res.latency_ms,
            total_retrieved=res.total_retrieved,
            confidence_score=res.confidence_score,
            is_low_confidence=res.is_low_confidence,
            warning_note=res.warning_note,
            hyde_passage=res.hyde_passage
        )
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationReport, summary="Query and evaluate output")
def evaluate_rag(req: QueryRequest):
    """Execute query and compute Faithfulness, Context Relevance, and Answer Relevance."""
    try:
        pipeline_inst = get_pipeline()
        evaluator_inst = get_evaluator()
        rag_res = pipeline_inst.query(question=req.question, top_k=req.top_k, use_hyde=req.use_hyde)
        eval_res = evaluator_inst.evaluate_response(rag_res, prefer_llm_judge=True)

        return EvaluationReport(
            query=rag_res.query,
            answer=rag_res.answer,
            context_relevance=MetricDetail(
                score=eval_res.context_relevance.score,
                rationale=eval_res.context_relevance.rationale
            ),
            faithfulness=MetricDetail(
                score=eval_res.faithfulness.score,
                rationale=eval_res.faithfulness.rationale
            ),
            answer_relevance=MetricDetail(
                score=eval_res.answer_relevance.score,
                rationale=eval_res.answer_relevance.rationale
            ),
            composite_score=eval_res.composite_score,
            evaluation_mode=eval_res.evaluation_mode
        )
    except Exception as e:
        logger.error(f"Error running evaluation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", summary="Reset vector store")
def reset_store():
    """Clear all stored vectors and indices."""
    try:
        get_vector_store().reset()
        return {"status": "success", "message": "Vector store collection reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
