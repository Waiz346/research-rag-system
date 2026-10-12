"""FastAPI entry point for the Research RAG Intelligence system."""
import io
import re
import time
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from src.config import settings
from src.ingestion.arxiv_collector import ArxivCollector
from src.ingestion.chunker import DocumentChunker
from src.indexing.vector_store import VectorStoreManager
from src.indexing.document_catalog import DocumentCatalog
from src.rag.pipeline import RAGPipeline
from src.rag.generator import LLMGenerator
from src.evals.evaluator import RAGEvaluator
from src.ingestion.gdrive_connector import get_gdrive_connector
from src.api.schemas import (
    IngestRequest, IngestResponse, PaperSummaryInfo,
    QueryRequest, QueryResponse, CitationSchema,
    DocumentSummarySchema, DocumentDetailSchema, UploadResponse,
    EvaluationReport, MetricDetail,
    HealthResponse,
    LLMConfigUpdate, LLMConfigResponse, LLMTestResponse,
    GDriveConnectRequest, GDriveStatusResponse, GDriveBrowseResponse,
    GDriveImportRequest, GDriveImportResponse, GDriveItem
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# State containers
vector_store = None
generator = None
pipeline = None
evaluator = None
document_catalog = None

def get_vector_store() -> VectorStoreManager:
    global vector_store
    if vector_store is None:
        vector_store = VectorStoreManager()
    return vector_store

def get_generator() -> LLMGenerator:
    global generator
    if generator is None:
        generator = LLMGenerator()
    return generator

def get_document_catalog() -> DocumentCatalog:
    global document_catalog
    if document_catalog is None:
        document_catalog = DocumentCatalog()
        # Sync existing papers from ChromaDB metadata into manifest if manifest is empty
        document_catalog.sync_from_chroma(get_vector_store().collection)
    return document_catalog

def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(vector_store=get_vector_store(), generator=get_generator())
    return pipeline

def get_evaluator() -> RAGEvaluator:
    global evaluator
    if evaluator is None:
        evaluator = RAGEvaluator(generator=get_generator())
    return evaluator

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook to initialize pipeline components on startup."""
    logger.info("Initializing Vector Store, Generator, and RAG pipeline...")
    get_vector_store()
    get_generator()
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
    cfg = get_generator().get_config()
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        llm_provider=f"{cfg['provider']} ({cfg['model_name']})",
        indexed_documents_count=get_vector_store().count()
    )

@app.get("/config/llm", response_model=LLMConfigResponse, summary="Get active LLM settings")
def get_llm_config():
    """Retrieve active LLM provider, model, and endpoint settings."""
    cfg = get_generator().get_config()
    return LLMConfigResponse(**cfg)

@app.post("/config/llm", response_model=LLMConfigResponse, summary="Switch LLM dynamically at runtime")
def update_llm_config(req: LLMConfigUpdate):
    """Switch LLM provider and credentials on-the-fly without server restart."""
    cfg = get_generator().update_config(
        provider=req.provider,
        model_name=req.model_name,
        api_key=req.api_key,
        base_url=req.base_url
    )
    return LLMConfigResponse(**cfg)

@app.post("/config/llm/test", response_model=LLMTestResponse, summary="Test LLM connection")
def test_llm_connection():
    """Probe the active LLM provider and measure latency/verify credentials."""
    result = get_generator().test_connection()
    return LLMTestResponse(**result)

@app.post("/ingest", response_model=IngestResponse, summary="Ingest papers from ArXiv")
def ingest_papers(req: IngestRequest):
    """Fetch ArXiv research papers, split into overlapping chunks, and index into ChromaDB."""
    try:
        collector = ArxivCollector()
        papers = collector.search_papers(
            query=req.query,
            seed_citation=req.seed_citation,
            narrow_keywords=req.narrow_keywords,
            title_keyword=req.title_keyword,
            abstract_keyword=req.abstract_keyword,
            min_citations=req.min_citations,
            max_results=req.max_papers,
            download_pdf=req.download_pdf
        )

        if not papers:
            raise HTTPException(
                status_code=404,
                detail="No papers found matching the specified citation reference and keyword filters."
            )

        chunker = DocumentChunker()
        all_chunks = []
        titles = []
        paper_summaries = []

        for paper in papers:
            cites_info = f" [Cites {paper.cites_seed}]" if paper.cites_seed else ""
            titles.append(f"{paper.title} ({paper.citation_count:,} citations{cites_info})")
            paper_summaries.append(PaperSummaryInfo(
                id=paper.id,
                title=paper.title,
                citation_count=paper.citation_count
            ))
            base_meta = {
                "doc_id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "pdf_url": paper.pdf_url,
                "citation_count": paper.citation_count,
                "cites_seed": paper.cites_seed or ""
            }
            chunks = chunker.split_text(
                paper.text_content,
                base_metadata=base_meta,
                section_aware=req.section_aware,
                exclude_references=req.exclude_references
            )
            all_chunks.extend(chunks)

        store = get_vector_store()
        indexed_count = store.add_chunks(all_chunks)

        # Register papers in lightweight disk catalog
        catalog = get_document_catalog()
        for paper in papers:
            p_chunks = [c for c in all_chunks if c.metadata.get("doc_id") == paper.id]
            catalog.register_document(
                doc_id=paper.id,
                title=paper.title,
                full_text=paper.text_content,
                source_type="arxiv",
                filename=f"{paper.id}.pdf",
                authors=", ".join(paper.authors),
                citation_count=paper.citation_count,
                chunk_count=len(p_chunks),
                sections=[c.section for c in p_chunks]
            )

        return IngestResponse(
            status="success",
            papers_fetched=len(papers),
            chunks_indexed=indexed_count,
            total_in_index=store.count(),
            paper_titles=titles,
            papers=paper_summaries
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload", response_model=UploadResponse, summary="Upload custom PDF, TXT, or Markdown document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index user files directly into the knowledge base (NotebookLM style)."""
    try:
        filename = file.filename or "uploaded_document.txt"
        ext = Path(filename).suffix.lower()
        if ext not in [".pdf", ".txt", ".md"]:
            raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")

        contents = await file.read()
        text_content = ""
        if ext == ".pdf":
            try:
                reader = PdfReader(io.BytesIO(contents))
                extracted = []
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        extracted.append(txt)
                text_content = "\n\n".join(extracted)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")
        else:
            text_content = contents.decode("utf-8", errors="replace")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty or contains no extractable text.")

        raw_stem = Path(filename).stem
        clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_stem)
        doc_id = f"doc_{clean_stem}_{int(time.time())}"
        title = raw_stem.replace("_", " ").title()

        chunker = DocumentChunker()
        base_meta = {
            "doc_id": doc_id,
            "title": title,
            "authors": "User Upload",
            "published": datetime.now().strftime("%Y-%m-%d"),
            "pdf_url": "",
            "citation_count": 0,
            "source_type": ext.lstrip(".")
        }

        chunks = chunker.split_text(text_content, base_metadata=base_meta, section_aware=True, exclude_references=True)
        store = get_vector_store()
        indexed_count = store.add_chunks(chunks)

        catalog = get_document_catalog()
        sections = [c.section for c in chunks]
        catalog.register_document(
            doc_id=doc_id,
            title=title,
            full_text=text_content,
            source_type=ext.lstrip("."),
            filename=filename,
            authors="User Upload",
            citation_count=0,
            chunk_count=len(chunks),
            sections=sections
        )

        return UploadResponse(
            status="success",
            doc_id=doc_id,
            title=title,
            chunks_indexed=indexed_count,
            total_in_index=store.count(),
            sections=list(dict.fromkeys(sections)),
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentSummarySchema], summary="List all indexed papers (Lightweight Manifest)")
def list_documents():
    """Memory-efficient document listing: loads only lightweight metadata, not full text."""
    catalog = get_document_catalog()
    summaries = catalog.list_documents()
    return [DocumentSummarySchema(**asdict(s)) for s in summaries]

@app.get("/documents/{doc_id}", response_model=DocumentDetailSchema, summary="Get full document text on-demand (Lazy Loaded)")
def get_document_detail(doc_id: str):
    """On-demand full content loading for document inspection."""
    catalog = get_document_catalog()
    doc = catalog.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentDetailSchema(
        summary=DocumentSummarySchema(**asdict(doc.summary)),
        full_text=doc.full_text
    )

@app.delete("/documents/{doc_id}", summary="Delete document from index and catalog")
def delete_document(doc_id: str):
    """Delete a paper from the document catalog and vector store."""
    catalog = get_document_catalog()
    deleted = catalog.delete_document(doc_id)
    get_vector_store().delete_by_doc_id(doc_id)
    return {"status": "success", "deleted": deleted, "doc_id": doc_id}

@app.post("/query", response_model=QueryResponse, summary="Query RAG pipeline")
def query_rag(req: QueryRequest):
    """Perform similarity search and synthesize answer with source citations."""
    try:
        pipeline_inst = get_pipeline()
        hist = [{"role": m.role, "content": m.content} for m in req.history] if req.history else None
        res = pipeline_inst.query(
            question=req.question,
            top_k=req.top_k,
            use_hyde=req.use_hyde,
            search_mode=req.search_mode,
            score_threshold=req.score_threshold,
            hybrid_alpha=req.hybrid_alpha,
            history=hist,
            doc_ids=req.doc_ids
        )
        
        citations = [
            CitationSchema(
                index=s.index,
                paper_id=s.paper_id,
                title=s.title,
                snippet=s.snippet,
                score=s.score,
                citation_count=s.citation_count,
                section=s.section,
                dense_score=s.dense_score,
                bm25_score=s.bm25_score,
                search_type=s.search_type,
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
            hyde_passage=res.hyde_passage,
            search_mode=res.search_mode,
            filtered_out_count=res.filtered_out_count,
            score_threshold=res.score_threshold
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

# ==============================================================================
# Google Drive Connector Endpoints
# ==============================================================================

@app.get("/gdrive/status", response_model=GDriveStatusResponse, summary="Get Google Drive connector status")
def gdrive_status():
    """Check whether Google Drive connector is active and its current auth mode."""
    conn = get_gdrive_connector()
    status = conn.get_status()
    return GDriveStatusResponse(
        connected=status["connected"],
        auth_type=status["auth_type"],
        account_email=status.get("account_email"),
        message=status["message"]
    )

@app.post("/gdrive/connect", response_model=GDriveStatusResponse, summary="Connect Google Drive with credentials")
def gdrive_connect(req: GDriveConnectRequest):
    """Authenticate Google Drive connector via Service Account, OAuth token, API key, gdrive CLI, or Demo mode."""
    try:
        conn = get_gdrive_connector()
        res = conn.connect(
            auth_type=req.auth_type,
            api_key=req.api_key,
            access_token=req.access_token,
            service_account_json=req.service_account_json
        )
        return GDriveStatusResponse(
            connected=True,
            auth_type=conn.auth_type,
            account_email=conn.account_email,
            message=res.get("message", "Connected successfully.")
        )
    except Exception as e:
        logger.error(f"Error connecting to Google Drive: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/gdrive/browse", response_model=GDriveBrowseResponse, summary="Browse Google Drive folder contents")
def gdrive_browse(folder_id: str = "root"):
    """List subfolders and files in a Google Drive folder."""
    try:
        conn = get_gdrive_connector()
        data = conn.list_folder_contents(folder_id=folder_id)
        return GDriveBrowseResponse(
            current_folder_id=data["current_folder_id"],
            current_folder_name=data["current_folder_name"],
            parent_id=data.get("parent_id"),
            folders=[GDriveItem(**f) for f in data.get("folders", [])],
            files=[GDriveItem(**f) for f in data.get("files", [])],
            total_items=data.get("total_items", 0)
        )
    except Exception as e:
        logger.error(f"Error browsing Google Drive folder {folder_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gdrive/import", response_model=GDriveImportResponse, summary="Import and index a file from Google Drive")
def gdrive_import(req: GDriveImportRequest):
    """Download a file from Google Drive, perform section chunking, and index into ChromaDB + BM25."""
    try:
        conn = get_gdrive_connector()
        filename, text_content, mime_type = conn.download_file_content(req.file_id)

        if not text_content.strip():
            raise HTTPException(status_code=400, detail=f"File {filename} is empty or contains no extractable text.")

        raw_stem = Path(filename).stem
        clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_stem)
        doc_id = f"gdrive_{clean_stem}_{int(time.time())}"
        title = req.title or raw_stem.replace("_", " ").title()

        chunker = DocumentChunker()
        base_meta = {
            "doc_id": doc_id,
            "title": title,
            "authors": "Google Drive",
            "published": datetime.now().strftime("%Y-%m-%d"),
            "pdf_url": "",
            "citation_count": 0,
            "source_type": "gdrive"
        }

        chunks = chunker.split_text(
            text_content,
            base_metadata=base_meta,
            section_aware=req.section_aware,
            exclude_references=req.exclude_references
        )
        store = get_vector_store()
        indexed_count = store.add_chunks(chunks)

        catalog = get_document_catalog()
        sections = [c.section for c in chunks]
        catalog.register_document(
            doc_id=doc_id,
            title=title,
            full_text=text_content,
            source_type="gdrive",
            filename=filename,
            authors="Google Drive",
            citation_count=0,
            chunk_count=len(chunks),
            sections=sections
        )

        return GDriveImportResponse(
            status="success",
            doc_id=doc_id,
            title=title,
            filename=filename,
            chunks_indexed=indexed_count,
            total_in_index=store.count(),
            sections=list(dict.fromkeys(sections)),
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing Google Drive file {req.file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

