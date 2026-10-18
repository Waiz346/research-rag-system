"""FastAPI entry point for the Research RAG Intelligence system."""
import io
import json
import re
import time
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pypdf import PdfReader

from src.config import settings
from src.ingestion.arxiv_collector import ArxivCollector
from src.ingestion.chunker import DocumentChunker
from src.ingestion.universal_parser import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    compute_deterministic_doc_id,
    parse_any_file,
    extract_from_url
)
from src.indexing.vector_store import VectorStoreManager
from src.indexing.document_catalog import DocumentCatalog, _extract_source_guide
from src.rag.pipeline import RAGPipeline
from src.rag.generator import LLMGenerator
from src.rag.studio_tools import generate_studio_artifact, format_markdown_export, generate_notebook_guide
from src.evals.evaluator import RAGEvaluator
from src.ingestion.gdrive_connector import get_gdrive_connector
from src.ingestion.classroom_connector import get_classroom_connector, save_oauth_credentials
from src.storage.workspace_store import get_workspace_store, WorkspaceStore
from src.api.schemas import (
    IngestRequest, IngestResponse, PaperSummaryInfo,
    QueryRequest, QueryResponse, CitationSchema,
    DocumentSummarySchema, DocumentDetailSchema, UploadResponse,
    EvaluationReport, MetricDetail,
    HealthResponse,
    LLMConfigUpdate, LLMConfigResponse, LLMTestResponse,
    GDriveConnectRequest, GDriveStatusResponse, GDriveBrowseResponse, GDriveSearchResponse,
    GDriveImportRequest, GDriveImportResponse, GDriveItem,
    GDriveAccountInfo, GDriveAccountsResponse, GDriveSwitchAccountRequest, GDriveLoginStatusResponse,
    GDriveBatchImportRequest, GDriveBatchImportResponse, BatchDeleteRequest, BatchDeleteResponse,
    WorkspaceCreate, WorkspaceDetail, WorkspaceSummary, WorkspaceListResponse,
    ConversationCreate, ConversationUpdate, ConversationSummary, ConversationDetail,
    MessagePayload, MessageResponse,
    ClassroomStatusResponse, ClassroomConnectRequest, ClassroomOAuthConfigRequest, ClassroomCoursesResponse,
    ClassroomCourseItem, ClassroomCourseStreamResponse, ClassroomAnnouncementItem,
    ClassroomCourseworkMaterialItem, ClassroomCourseworkItem, ClassroomMaterialItem,
    ClassroomImportRequest, ClassroomImportResponse, ClassroomBatchImportRequest,
    ClassroomBatchImportResponse, ClassroomAccountsResponse, ClassroomAccountInfo,
    ClassroomSwitchAccountRequest, ClassroomLoginStatusResponse,
    UrlIngestRequest, NoteIngestRequest, DeduplicateResponse,
    StudioGenerateRequest, StudioGenerateResponse,
    MarkdownExportRequest, MarkdownExportResponse, NotebookGuideResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve frontend UI path (compatible with both dev and PyInstaller frozen bundle)
import sys
if getattr(sys, "frozen", False):
    FRONTEND_DIR = Path(sys._MEIPASS) / "frontend"
else:
    FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

# State containers
vector_store = None
generator = None
pipeline = None
evaluator = None
document_catalog = None
workspace_store = None

def get_workspace_store_instance() -> WorkspaceStore:
    global workspace_store
    if workspace_store is None:
        workspace_store = get_workspace_store()
    return workspace_store

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
    get_workspace_store_instance()
    yield
    logger.info("Shutting down application...")

app = FastAPI(
    title="NexusRAG Research Intelligence System",
    description="End-to-End ArXiv Paper Ingestion, Hybrid Semantic Retrieval, and Automated RAG Evaluation Studio",
    version="2.5.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def serve_workbench_ui():
    """Serve the standalone NexusRAG HTML5 Research Workbench."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse(
        content="<h1>NexusRAG Workbench</h1><p>Frontend index.html not found.</p>",
        status_code=404
    )

@app.get("/api/info", summary="API metadata endpoint")
def api_info():
    return {
        "message": "NexusRAG Research Intelligence System is online.",
        "ui": "/",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, summary="Check service health")
def health():
    cfg = get_generator().get_config()
    store = get_workspace_store_instance()
    db_engine = "PostgreSQL (Cloud Database)" if not store.resolved_url.startswith("sqlite") else "SQLite (Local Fallback)"
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        llm_provider=f"{cfg['provider']} ({cfg['model_name']})",
        indexed_documents_count=get_vector_store().count(),
        database_engine=db_engine
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
            err_msg = collector.last_error or "No papers found matching the specified citation reference and keyword filters. Try broadening your query or lowering the minimum citations."
            code = 429 if (collector.last_error and "429" in collector.last_error) else 404
            raise HTTPException(
                status_code=code,
                detail=err_msg
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
            target_ws = req.workspace_id or "default"
            base_meta = {
                "doc_id": paper.id,
                "workspace_id": target_ws,
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

        # Register papers in online database (and local catalog mirror)
        catalog = get_document_catalog()
        ws_store = get_workspace_store_instance()
        target_ws = req.workspace_id or "default"

        for paper in papers:
            p_chunks = [c for c in all_chunks if c.metadata.get("doc_id") == paper.id]
            authors_str = ", ".join(paper.authors) if isinstance(paper.authors, list) else str(paper.authors)
            sections_list = [c.section for c in p_chunks]

            ws_store.register_document(
                doc_id=paper.id,
                workspace_id=target_ws,
                title=paper.title,
                full_text=paper.text_content,
                source_type="arxiv",
                filename=f"{paper.id}.pdf",
                authors=authors_str,
                citation_count=paper.citation_count,
                chunk_count=len(p_chunks),
                sections=sections_list
            )
            catalog.register_document(
                doc_id=paper.id,
                title=paper.title,
                full_text=paper.text_content,
                source_type="arxiv",
                filename=f"{paper.id}.pdf",
                authors=authors_str,
                citation_count=paper.citation_count,
                chunk_count=len(p_chunks),
                sections=sections_list
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

def _index_and_register_document(
    doc_id: str,
    workspace_id: str,
    title: str,
    text_content: str,
    source_type: str,
    filename: str,
    authors: str = "User Upload",
    pdf_url: str = "",
    section_aware: bool = True,
    exclude_references: bool = True
) -> tuple[int, List[str]]:
    """Index document chunks (replacing any prior chunks for the same doc_id) and register in DB + catalog."""
    chunker = DocumentChunker()
    base_meta = {
        "doc_id": doc_id,
        "workspace_id": workspace_id,
        "title": title,
        "authors": authors,
        "published": datetime.now().strftime("%Y-%m-%d"),
        "pdf_url": pdf_url,
        "citation_count": 0,
        "source_type": source_type
    }
    chunks = chunker.split_text(
        text_content,
        base_metadata=base_meta,
        section_aware=section_aware,
        exclude_references=exclude_references
    )
    store = get_vector_store()
    # Clean up old chunks if re-indexing the same deterministic doc_id
    store.delete_by_doc_id(doc_id)
    indexed_count = store.add_chunks(chunks)
    sections = [c.section for c in chunks]

    ws_store = get_workspace_store_instance()
    ws_store.register_document(
        doc_id=doc_id,
        workspace_id=workspace_id,
        title=title,
        full_text=text_content,
        source_type=source_type,
        filename=filename,
        authors=authors,
        citation_count=0,
        chunk_count=len(chunks),
        sections=sections
    )

    catalog = get_document_catalog()
    catalog.register_document(
        doc_id=doc_id,
        title=title,
        full_text=text_content,
        source_type=source_type,
        filename=filename,
        authors=authors,
        citation_count=0,
        chunk_count=len(chunks),
        sections=sections
    )
    return indexed_count, sections


@app.post("/documents/upload", response_model=UploadResponse, summary="Upload PDF, Office (.docx/.pptx/.xlsx), Data (.csv/.ipynb), Code, Image, or Text document")
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: str = Form("default")
):
    """Upload and index everyday user files directly into the online database under a workspace."""
    try:
        filename = file.filename or "uploaded_document.txt"
        ext = Path(filename).suffix.lower()
        if ext and ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join( sorted(SUPPORTED_UPLOAD_EXTENSIONS) )}"
            )

        contents = await file.read()
        gen = get_generator()
        try:
            text_content, source_type = parse_any_file(
                filename=filename,
                raw_bytes=contents,
                mime_type=file.content_type or "",
                vision_ocr_fn=gen.extract_vision_text
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse {filename}: {e}")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty or contains no extractable text.")

        raw_stem = Path(filename).stem
        title = raw_stem.replace("_", " ").replace("-", " ").strip().title() or filename
        doc_id = compute_deterministic_doc_id("doc", filename, text_content)

        indexed_count, sections = _index_and_register_document(
            doc_id=doc_id,
            workspace_id=workspace_id,
            title=title,
            text_content=text_content,
            source_type=source_type,
            filename=filename,
            authors="User Upload",
            section_aware=True,
            exclude_references=(source_type in {"pdf", "docx"})
        )

        return UploadResponse(
            status="success",
            doc_id=doc_id,
            title=title,
            chunks_indexed=indexed_count,
            total_in_index=get_vector_store().count(),
            sections=list(dict.fromkeys(sections)),
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/ingest-url", response_model=UploadResponse, summary="Ingest Web Article, YouTube Video Transcript, or PDF URL")
def ingest_url_document(req: UrlIngestRequest):
    """Fetch and index a web page article, YouTube transcript, or direct PDF link."""
    try:
        gen = get_generator()
        try:
            auto_title, text_content, source_type = extract_from_url(
                req.url,
                vision_ocr_fn=gen.extract_vision_text
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        if not text_content or not text_content.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable content from URL.")

        title = (req.title or auto_title or req.url).strip()
        workspace_id = req.workspace_id or "default"
        doc_id = compute_deterministic_doc_id(source_type, req.url, req.url)
        filename = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', title)[:50]}.md"

        indexed_count, sections = _index_and_register_document(
            doc_id=doc_id,
            workspace_id=workspace_id,
            title=title,
            text_content=text_content,
            source_type=source_type,
            filename=filename,
            authors="YouTube Transcript" if source_type == "youtube" else "Web Article",
            pdf_url=req.url,
            section_aware=req.section_aware,
            exclude_references=req.exclude_references
        )

        return UploadResponse(
            status="success",
            doc_id=doc_id,
            title=title,
            chunks_indexed=indexed_count,
            total_in_index=get_vector_store().count(),
            sections=list(dict.fromkeys(sections)),
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting URL {req.url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/ingest-note", response_model=UploadResponse, summary="Ingest a quick text/markdown note or pasted snippet")
def ingest_note_document(req: NoteIngestRequest):
    """Save and index a quick note, lecture summary, or pasted text directly into the workspace."""
    try:
        text_content = req.content.strip()
        if not text_content:
            raise HTTPException(status_code=400, detail="Note content cannot be empty.")

        title = req.title.strip()
        workspace_id = req.workspace_id or "default"
        doc_id = compute_deterministic_doc_id("note", title, text_content)
        filename = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', title)[:50]}.md"

        indexed_count, sections = _index_and_register_document(
            doc_id=doc_id,
            workspace_id=workspace_id,
            title=title,
            text_content=text_content,
            source_type="note",
            filename=filename,
            authors="Quick Note",
            section_aware=True,
            exclude_references=False
        )

        return UploadResponse(
            status="success",
            doc_id=doc_id,
            title=title,
            chunks_indexed=indexed_count,
            total_in_index=get_vector_store().count(),
            sections=list(dict.fromkeys(sections)),
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting quick note: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/deduplicate", response_model=DeduplicateResponse, summary="Remove duplicate documents from workspace and vector store")
def deduplicate_documents(workspace_id: Optional[str] = None):
    """Scan indexed documents for duplicate titles/content and remove redundant copies."""
    ws_store = get_workspace_store_instance()
    catalog = get_document_catalog()
    vstore = get_vector_store()

    docs = ws_store.list_documents(workspace_id=workspace_id)
    seen_keys = {}
    removed_ids = []

    # Sort newest first so we keep the most recently updated copy
    sorted_docs = sorted(
        docs,
        key=lambda d: d.get("created_at") or d.get("uploaded_at") or "",
        reverse=True
    )
    for d in sorted_docs:
        doc_id = d["doc_id"]
        ws = d.get("workspace_id", "default")
        norm_title = re.sub(r"\s+", " ", (d.get("title") or "").strip().lower())
        char_bucket = d.get("char_count", 0)
        dedup_key = (ws, norm_title, char_bucket)
        if dedup_key in seen_keys:
            removed_ids.append(doc_id)
        else:
            seen_keys[dedup_key] = doc_id

    if removed_ids:
        ws_store.batch_delete_documents(removed_ids)
        for rid in removed_ids:
            try:
                catalog.delete_document(rid)
                vstore.delete_by_doc_id(rid)
            except Exception as e:
                logger.warning(f"Failed cleaning duplicate doc {rid}: {e}")

    # Also clean up duplicate entries in local DocumentCatalog that aren't in ws_store
    cat_docs = catalog.list_documents()
    cat_seen = {}
    for s in sorted(
        cat_docs,
        key=lambda x: getattr(x, "created_at", "") or getattr(x, "uploaded_at", "") or "",
        reverse=True
    ):
        key = (s.title.strip().lower(), s.char_count)
        if key in cat_seen and s.doc_id not in removed_ids:
            removed_ids.append(s.doc_id)
            catalog.delete_document(s.doc_id)
            vstore.delete_by_doc_id(s.doc_id)
        else:
            cat_seen[key] = s.doc_id

    remaining = len(ws_store.list_documents(workspace_id=workspace_id))
    return DeduplicateResponse(
        status="success",
        duplicates_removed=len(removed_ids),
        removed_doc_ids=removed_ids,
        remaining_documents=remaining
    )


@app.get("/documents", response_model=List[DocumentSummarySchema], summary="List all indexed papers from online database")
def list_documents(workspace_id: Optional[str] = None):
    """Memory-efficient document listing from online database, enriched with NotebookLM Source Guides."""
    catalog = get_document_catalog()
    cat_map = {s.doc_id: s for s in catalog.list_documents()}

    ws_store = get_workspace_store_instance()
    docs = ws_store.list_documents(workspace_id=workspace_id)

    # If querying the default workspace (or all), also include any catalog documents not yet in ws_store
    if workspace_id in (None, "default"):
        all_ws_docs = ws_store.list_documents(workspace_id=None) or []
        known_ws_ids = {d.get("doc_id") for d in all_ws_docs}
        docs = list(docs or [])
        for doc_id, cat_item in cat_map.items():
            if doc_id not in known_ws_ids:
                d_dict = asdict(cat_item)
                d_dict["workspace_id"] = "default"
                docs.append(d_dict)

    if docs:
        enriched = []
        for d in docs:
            doc_id = d.get("doc_id", "")
            cat_item = cat_map.get(doc_id)
            if cat_item:
                d["summary"] = cat_item.summary
                d["key_topics"] = cat_item.key_topics
                d["suggested_questions"] = cat_item.suggested_questions
            else:
                guide = _extract_source_guide(d.get("title", doc_id), "", d.get("sections", []))
                d.update(guide)
            enriched.append(DocumentSummarySchema(**d))
        return enriched

    if workspace_id in (None, "default"):
        return [DocumentSummarySchema(**asdict(s), workspace_id="default") for s in cat_map.values()]
    return []


@app.get("/notebook/guide", response_model=NotebookGuideResponse, summary="Get NotebookLM-style Notebook Guide & Suggested Questions")
def get_notebook_guide(
    workspace_id: str = Query("default"),
    doc_ids: Optional[str] = Query(None, description="Comma-separated doc_ids if scoped")
):
    """Generate a NotebookLM-style overview, key topics, and suggested questions for active workspace sources."""
    all_docs = [d.model_dump() for d in list_documents(workspace_id=workspace_id)]
    if doc_ids:
        wanted = {x.strip() for x in doc_ids.split(",") if x.strip()}
        if wanted:
            all_docs = [d for d in all_docs if d["doc_id"] in wanted]
    guide = generate_notebook_guide(all_docs)
    return NotebookGuideResponse(
        workspace_id=workspace_id,
        source_count=guide["source_count"],
        overview=guide["overview"],
        key_topics=guide["key_topics"],
        suggested_questions=guide["suggested_questions"]
    )


@app.get("/documents/{doc_id}", response_model=DocumentDetailSchema, summary="Get full document text on-demand from online database")
def get_document_detail(doc_id: str):
    """On-demand full content and Source Guide loading from online database for document inspection."""
    ws_store = get_workspace_store_instance()
    doc_dict = ws_store.get_document(doc_id)
    if doc_dict:
        full_text = doc_dict.pop("full_text", "")
        guide = _extract_source_guide(doc_dict.get("title", doc_id), full_text[:3500], doc_dict.get("sections", []))
        doc_dict.update(guide)
        return DocumentDetailSchema(
            summary=DocumentSummarySchema(**doc_dict),
            full_text=full_text
        )

    catalog = get_document_catalog()
    doc = catalog.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentDetailSchema(
        summary=DocumentSummarySchema(**asdict(doc.summary), workspace_id="default"),
        full_text=doc.full_text
    )

@app.delete("/documents/{doc_id}", summary="Delete document from online database and index")
def delete_document(doc_id: str):
    """Delete a paper from the online database, catalog, and vector store."""
    ws_store = get_workspace_store_instance()
    deleted = ws_store.delete_document(doc_id)
    get_document_catalog().delete_document(doc_id)
    get_vector_store().delete_by_doc_id(doc_id)
    return {"status": "success", "deleted": deleted, "doc_id": doc_id}

@app.post("/documents/batch-delete", response_model=BatchDeleteResponse, summary="Batch delete documents from online database")
def batch_delete_documents(req: BatchDeleteRequest):
    """Delete multiple documents from online database in a single request."""
    ws_store = get_workspace_store_instance()
    catalog = get_document_catalog()
    store = get_vector_store()
    ws_store.batch_delete_documents(req.doc_ids)

    deleted_ids = []
    for doc_id in req.doc_ids:
        try:
            catalog.delete_document(doc_id)
            store.delete_by_doc_id(doc_id)
            deleted_ids.append(doc_id)
        except Exception as e:
            logger.warning(f"Error deleting doc {doc_id}: {e}")
    return BatchDeleteResponse(
        deleted_count=len(deleted_ids),
        deleted_ids=deleted_ids
    )

@app.post("/query", response_model=QueryResponse, summary="Query RAG pipeline")
def query_rag(req: QueryRequest):
    """Perform similarity search and synthesize answer with source citations."""
    try:
        pipeline_inst = get_pipeline()
        hist = [{"role": m.role, "content": m.content} for m in req.history] if req.history else None

        # Workspace scoping
        effective_doc_ids = req.doc_ids
        if not effective_doc_ids and req.workspace_id:
            ws_docs = list_documents(workspace_id=req.workspace_id)
            if ws_docs:
                effective_doc_ids = [d.doc_id for d in ws_docs]
            else:
                effective_doc_ids = ["__EMPTY_WORKSPACE__"]

        res = pipeline_inst.query(
            question=req.question,
            top_k=req.top_k,
            use_hyde=req.use_hyde,
            search_mode=req.search_mode,
            score_threshold=req.score_threshold,
            hybrid_alpha=req.hybrid_alpha,
            history=hist,
            doc_ids=effective_doc_ids
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
            rewritten_query=res.rewritten_query,
            query_intent=res.query_intent,
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
            score_threshold=res.score_threshold,
            follow_up_questions=res.follow_up_questions
        )
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", summary="Stream RAG query response via Server-Sent Events (SSE)")
def query_rag_stream(req: QueryRequest):
    """Execute retrieval and stream citations + answer tokens progressively using SSE."""
    def event_stream():
        try:
            res = query_rag(req)
            meta_payload = {
                "query": res.query,
                "rewritten_query": res.rewritten_query,
                "query_intent": res.query_intent,
                "sources": [s.model_dump() for s in res.sources],
                "latency_ms": res.latency_ms,
                "total_retrieved": res.total_retrieved,
                "confidence_score": res.confidence_score,
                "is_low_confidence": res.is_low_confidence,
                "warning_note": res.warning_note,
                "hyde_passage": res.hyde_passage,
                "search_mode": res.search_mode,
                "filtered_out_count": res.filtered_out_count,
                "score_threshold": res.score_threshold,
            }
            yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

            # Stream answer in smooth word chunks so UI renders progressively
            words = re.split(r"(\s+)", res.answer or "")
            chunk_buf = ""
            for w in words:
                chunk_buf += w
                if len(chunk_buf) >= 18:
                    yield f"event: token\ndata: {json.dumps({'token': chunk_buf})}\n\n"
                    chunk_buf = ""
            if chunk_buf:
                yield f"event: token\ndata: {json.dumps({'token': chunk_buf})}\n\n"

            yield f"event: done\ndata: {json.dumps({'answer': res.answer})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/studio/generate", response_model=StudioGenerateResponse, summary="Generate Study Guide, Quiz, Comparison Matrix, Action Items, or Audio Briefing")
def studio_generate_endpoint(req: StudioGenerateRequest):
    """Generate 1-click study and research artifacts from workspace documents."""
    effective_action = req.tool_type or req.action_type or "study_guide"
    try:
        t0 = time.perf_counter()
        ws_store = get_workspace_store_instance()
        catalog = get_document_catalog()
        target_docs = []

        doc_ids_to_load = req.doc_ids
        if not doc_ids_to_load:
            ws_id = req.workspace_id or "default"
            summaries = list_documents(workspace_id=ws_id)
            doc_ids_to_load = [s.doc_id for s in summaries[:12]]

        for did in (doc_ids_to_load or []):
            doc = ws_store.get_document(did)
            if doc:
                target_docs.append(doc)
            else:
                cat_doc = catalog.get_document(did)
                if cat_doc:
                    target_docs.append({
                        "doc_id": cat_doc.summary.doc_id,
                        "title": cat_doc.summary.title,
                        "authors": cat_doc.summary.authors,
                        "source_type": cat_doc.summary.source_type,
                        "sections": cat_doc.summary.sections,
                        "full_text": cat_doc.full_text
                    })

        result = generate_studio_artifact(
            action_type=effective_action,
            documents=target_docs,
            generator=get_generator(),
            custom_focus=req.custom_focus or ""
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return StudioGenerateResponse(
            tool_type=result["action_type"],
            action_type=result["action_type"],
            title=result["title"],
            content=result["content"],
            doc_count=len(target_docs),
            sources=[CitationSchema(**s) for s in result.get("sources", [])],
            latency_ms=round(elapsed_ms, 2)
        )
    except Exception as e:
        logger.error(f"Error generating studio artifact ({effective_action}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/markdown", response_model=MarkdownExportResponse, summary="Export conversation or studio report as Obsidian Markdown + BibTeX")
def export_markdown_endpoint(req: MarkdownExportRequest):
    """Format research synthesis or study artifacts as downloadable Obsidian Markdown with YAML frontmatter and BibTeX."""
    try:
        body_text = req.content if req.content is not None else (req.answer or "")
        ws_label = req.workspace_name or req.workspace_id or "default"
        md_text = format_markdown_export(
            title=req.title,
            query=req.query,
            answer=body_text,
            sources=req.sources,
            workspace_id=ws_label,
            include_bibtex=req.include_bibtex,
            obsidian_compatible=req.obsidian_compatible
        )
        # Extract BibTeX block if present
        bib_match = re.search(r"```bibtex\s*(.*?)```", md_text, flags=re.DOTALL)
        bibtex_str = bib_match.group(1).strip() if bib_match else ""
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", req.title.strip())[:45].strip("_") or "nexusrag_note"
        return MarkdownExportResponse(
            markdown=md_text,
            bibtex=bibtex_str,
            filename=f"{slug}.md"
        )
    except Exception as e:
        logger.error(f"Error exporting markdown: {e}", exc_info=True)
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

@app.get("/gdrive/accounts", response_model=GDriveAccountsResponse, summary="List saved Google accounts")
def list_gdrive_accounts():
    """List all saved Google accounts available for quick switching."""
    try:
        conn = get_gdrive_connector()
        accounts = conn.list_saved_accounts()
        return GDriveAccountsResponse(
            accounts=[GDriveAccountInfo(**acc) for acc in accounts],
            active_account=conn.account_email if conn.auth_type == "user_account" else None,
            count=len(accounts)
        )
    except Exception as e:
        logger.error(f"Error listing Google Drive accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gdrive/accounts/switch", summary="Quick switch active Google account")
def switch_gdrive_account(req: GDriveSwitchAccountRequest):
    """Instantly switch active Google Drive session to a saved account."""
    try:
        conn = get_gdrive_connector()
        res = conn.switch_account(req.email)
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error switching Google Drive account: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/gdrive/accounts/login", response_model=GDriveLoginStatusResponse, summary="Launch Google Account Chooser in browser")
def login_gdrive_account():
    """Launch Google OAuth flow in user's browser with account selection prompt."""
    try:
        conn = get_gdrive_connector()
        res = conn.start_browser_login()
        return GDriveLoginStatusResponse(**res)
    except Exception as e:
        logger.error(f"Error initiating Google sign-in: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/gdrive/accounts/login-status", response_model=GDriveLoginStatusResponse, summary="Check status of browser sign-in")
def get_gdrive_login_status():
    """Check whether browser-based OAuth flow completed or is still waiting."""
    try:
        conn = get_gdrive_connector()
        res = conn.get_login_status()
        return GDriveLoginStatusResponse(**res)
    except Exception as e:
        logger.error(f"Error checking sign-in status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/gdrive/accounts/{email}", summary="Remove a saved Google account")
def delete_gdrive_account(email: str):
    """Remove credentials for a saved Google account."""
    try:
        conn = get_gdrive_connector()
        return conn.delete_saved_account(email)
    except Exception as e:
        logger.error(f"Error deleting saved account {email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/gdrive/search", response_model=GDriveSearchResponse, summary="Search files and folders in Google Drive")
def gdrive_search(query: str = Query(..., min_length=1, description="Keywords or filename to search")):
    """Search Google Drive by file/folder name or content."""
    try:
        conn = get_gdrive_connector()
        data = conn.search_drive(query)
        return GDriveSearchResponse(
            query=query,
            folders=[GDriveItem(**f) for f in data.get("folders", [])],
            files=[GDriveItem(**f) for f in data.get("files", [])],
            total_items=data.get("total_items", 0)
        )
    except Exception as e:
        logger.error(f"Error searching Google Drive for '{query}': {e}")
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
        title = req.title or raw_stem.replace("_", " ").title()
        ws_id = req.workspace_id or "default"
        doc_id = compute_deterministic_doc_id("gdrive", filename, f"{req.file_id}:{ws_id}")

        indexed_count, sections = _index_and_register_document(
            doc_id=doc_id,
            workspace_id=ws_id,
            title=title,
            text_content=text_content,
            source_type="gdrive",
            filename=filename,
            authors="Google Drive",
            section_aware=req.section_aware,
            exclude_references=req.exclude_references
        )

        return GDriveImportResponse(
            status="success",
            doc_id=doc_id,
            title=title,
            filename=filename,
            chunks_indexed=indexed_count,
            total_in_index=get_vector_store().count(),
            sections=list(dict.fromkeys(sections)),
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing Google Drive file {req.file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gdrive/import/batch", response_model=GDriveBatchImportResponse, summary="Batch import and index multiple files from Google Drive")
def gdrive_import_batch(req: GDriveBatchImportRequest):
    """Batch download and index multiple files from Google Drive into online database."""
    conn = get_gdrive_connector()
    ws_id = req.workspace_id or "default"

    imported_docs = []
    errors = []

    for item in req.items:
        try:
            filename, text_content, mime_type = conn.download_file_content(item.file_id)
            if not text_content.strip():
                errors.append({"file_id": item.file_id, "error": f"File {filename} is empty or unparseable."})
                continue

            raw_stem = Path(filename).stem
            title = item.title or raw_stem.replace("_", " ").title()
            doc_id = compute_deterministic_doc_id("gdrive", filename, f"{item.file_id}:{ws_id}")

            indexed_count, _ = _index_and_register_document(
                doc_id=doc_id,
                workspace_id=ws_id,
                title=title,
                text_content=text_content,
                source_type="gdrive",
                filename=filename,
                authors="Google Drive",
                section_aware=req.section_aware,
                exclude_references=req.exclude_references
            )

            imported_docs.append({
                "doc_id": doc_id,
                "title": title,
                "filename": filename,
                "chunks_indexed": indexed_count
            })
        except Exception as e:
            logger.warning(f"Error importing batch file {item.file_id}: {e}")
            errors.append({"file_id": item.file_id, "error": str(e)})

    return GDriveBatchImportResponse(
        total_requested=len(req.items),
        total_succeeded=len(imported_docs),
        total_failed=len(errors),
        imported_docs=imported_docs,
        errors=errors
    )

# ==============================================================================
# Google Classroom Connector Endpoints
# ==============================================================================

@app.get("/classroom/status", response_model=ClassroomStatusResponse, summary="Get Google Classroom connector status")
def classroom_status():
    """Check Google Classroom connector state and connection validation."""
    conn = get_classroom_connector()
    val = conn.validate_connection()
    return ClassroomStatusResponse(
        connected=val.get("valid", False),
        auth_type=val.get("auth_type", "none"),
        account_email=val.get("account_email"),
        courses_count=val.get("courses_count", 0),
        message=val.get("message", ""),
        latency_ms=val.get("latency_ms"),
        valid=val.get("valid", False),
        error=val.get("error")
    )

@app.post("/classroom/oauth-config", summary="Save Google Cloud OAuth client credentials for Classroom")
def save_classroom_oauth(req: ClassroomOAuthConfigRequest):
    """Save Google Cloud OAuth Client ID and Client Secret with Classroom API enabled."""
    save_oauth_credentials(req.client_id, req.client_secret)
    return {"status": "success", "message": "Google Cloud OAuth credentials saved."}

@app.post("/classroom/connect", response_model=ClassroomStatusResponse, summary="Connect Google Classroom")
def classroom_connect(req: ClassroomConnectRequest):
    """Connect to Google Classroom via access token or service account."""
    try:
        conn = get_classroom_connector()
        res = conn.connect(
            auth_type=req.auth_type,
            access_token=req.access_token,
            service_account_json=req.service_account_json
        )
        val = conn.validate_connection()
        return ClassroomStatusResponse(
            connected=val.get("valid", False),
            auth_type=conn.auth_type,
            account_email=conn.account_email,
            courses_count=val.get("courses_count", 0),
            message=res.get("message", "Connected successfully."),
            latency_ms=val.get("latency_ms"),
            valid=val.get("valid", False),
            error=val.get("error")
        )
    except Exception as e:
        logger.error(f"Error connecting to Google Classroom: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/classroom/accounts", response_model=ClassroomAccountsResponse, summary="List saved Classroom accounts")
def list_classroom_accounts():
    """List all saved Google accounts with Classroom scopes."""
    try:
        conn = get_classroom_connector()
        accounts = conn.list_saved_accounts()
        return ClassroomAccountsResponse(
            accounts=[ClassroomAccountInfo(**a) for a in accounts],
            active_account=conn.account_email if conn.auth_type == "user_account" else None,
            count=len(accounts)
        )
    except Exception as e:
        logger.error(f"Error listing Classroom accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classroom/accounts/switch", summary="Switch active Classroom account")
def switch_classroom_account(req: ClassroomSwitchAccountRequest):
    """Switch active Google Classroom account session."""
    try:
        conn = get_classroom_connector()
        return conn.switch_account(req.email)
    except Exception as e:
        logger.error(f"Error switching Classroom account: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/classroom/accounts/login", response_model=ClassroomLoginStatusResponse, summary="Launch Google Classroom OAuth login")
def login_classroom_account(custom_client_id: Optional[str] = None, custom_client_secret: Optional[str] = None):
    """Launch Google OAuth login with Classroom and Drive permissions in default browser."""
    try:
        conn = get_classroom_connector()
        res = conn.start_browser_login(custom_client_id=custom_client_id, custom_client_secret=custom_client_secret)
        return ClassroomLoginStatusResponse(**res)
    except Exception as e:
        logger.error(f"Error initiating Classroom login: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/classroom/accounts/login-status", response_model=ClassroomLoginStatusResponse, summary="Check Classroom login status")
def get_classroom_login_status():
    """Check whether browser-based Classroom authorization completed."""
    try:
        conn = get_classroom_connector()
        res = conn.get_login_status()
        return ClassroomLoginStatusResponse(**res)
    except Exception as e:
        logger.error(f"Error checking Classroom login status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classroom/courses", response_model=ClassroomCoursesResponse, summary="List enrolled courses")
def list_classroom_courses():
    """List all enrolled and active courses in Google Classroom."""
    try:
        conn = get_classroom_connector()
        courses = conn.list_courses()
        return ClassroomCoursesResponse(
            courses=[ClassroomCourseItem(**c) for c in courses],
            total_courses=len(courses)
        )
    except Exception as e:
        logger.error(f"Error listing courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classroom/courses/{course_id}/stream", response_model=ClassroomCourseStreamResponse, summary="Get course announcements and materials")
def get_classroom_course_stream(course_id: str):
    """Retrieve full course stream: announcements, coursework materials, and assignments."""
    try:
        conn = get_classroom_connector()
        stream = conn.get_course_stream(course_id)
        return ClassroomCourseStreamResponse(
            course_id=stream["course_id"],
            course_name=stream["course_name"],
            announcements=[ClassroomAnnouncementItem(**a) for a in stream.get("announcements", [])],
            coursework_materials=[ClassroomCourseworkMaterialItem(**m) for m in stream.get("coursework_materials", [])],
            coursework=[ClassroomCourseworkItem(**c) for c in stream.get("coursework", [])],
            total_items=stream.get("total_items", 0)
        )
    except Exception as e:
        logger.error(f"Error loading course stream {course_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classroom/import", response_model=ClassroomImportResponse, summary="Import Classroom document into active workspace")
def classroom_import(req: ClassroomImportRequest):
    """Download and index a Classroom attachment or announcement into ChromaDB + BM25 and online DB."""
    try:
        conn = get_classroom_connector()
        filename, text_content, mime_type = conn.download_document_content(
            item_type=req.item_type,
            title=req.title,
            drive_file_id=req.drive_file_id,
            raw_text=req.raw_text,
            mime_type=req.mime_type
        )
        if not text_content.strip():
            raise HTTPException(status_code=400, detail=f"Content for '{req.title}' is empty or unextractable.")

        ws_id = req.workspace_id or "default"
        course_label = req.course_name or req.course_id
        doc_id = compute_deterministic_doc_id(
            "class",
            filename,
            f"{req.course_id}:{req.drive_file_id or req.title}:{ws_id}"
        )

        indexed_count, sections = _index_and_register_document(
            doc_id=doc_id,
            workspace_id=ws_id,
            title=req.title,
            text_content=text_content,
            source_type="classroom",
            filename=filename,
            authors=f"Google Classroom ({course_label})",
            section_aware=req.section_aware,
            exclude_references=req.exclude_references
        )

        return ClassroomImportResponse(
            status="success",
            doc_id=doc_id,
            title=req.title,
            course_name=course_label,
            chunks_indexed=indexed_count,
            total_in_index=get_vector_store().count(),
            sections=sections,
            char_count=len(text_content)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing Classroom document '{req.title}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classroom/import/batch", response_model=ClassroomBatchImportResponse, summary="Batch import multiple Classroom documents")
def classroom_import_batch(req: ClassroomBatchImportRequest):
    """Batch download and index multiple Classroom materials/announcements into online DB."""
    conn = get_classroom_connector()
    ws_id = req.workspace_id or "default"

    imported_docs = []
    errors = []

    for item in req.items:
        try:
            filename, text_content, mime_type = conn.download_document_content(
                item_type=item.item_type,
                title=item.title,
                drive_file_id=item.drive_file_id,
                raw_text=item.raw_text,
                mime_type=item.mime_type
            )
            if not text_content.strip():
                errors.append({"title": item.title, "error": "Empty or unparseable text content."})
                continue

            course_label = item.course_name or item.course_id
            doc_id = compute_deterministic_doc_id(
                "class",
                filename,
                f"{item.course_id}:{item.drive_file_id or item.title}:{ws_id}"
            )

            indexed_count, sections = _index_and_register_document(
                doc_id=doc_id,
                workspace_id=ws_id,
                title=item.title,
                text_content=text_content,
                source_type="classroom",
                filename=filename,
                authors=f"Google Classroom ({course_label})",
                section_aware=req.section_aware,
                exclude_references=req.exclude_references
            )
            imported_docs.append({
                "doc_id": doc_id,
                "title": item.title,
                "filename": filename,
                "chunks_indexed": indexed_count,
                "sections": sections
            })
        except Exception as e:
            logger.warning(f"Error importing batch item {item.title}: {e}")
            errors.append({"title": item.title, "error": str(e)})

    return ClassroomBatchImportResponse(
        total_requested=len(req.items),
        total_succeeded=len(imported_docs),
        total_failed=len(errors),
        imported_docs=imported_docs,
        errors=errors
    )



# ==============================================================================
# Workspace, Folder & Conversation Management Endpoints
# ==============================================================================

@app.get("/workspaces", response_model=WorkspaceListResponse, summary="List all workspaces")
def list_workspaces():
    """List all research workspaces / folders with their conversation threads."""
    store = get_workspace_store_instance()
    workspaces = store.list_workspaces()
    return WorkspaceListResponse(
        workspaces=workspaces,
        active_workspace_id=workspaces[0]["id"] if workspaces else None,
        count=len(workspaces)
    )

@app.post("/workspaces", response_model=WorkspaceDetail, summary="Create a new workspace")
def create_workspace(req: WorkspaceCreate):
    """Create a new workspace / folder."""
    store = get_workspace_store_instance()
    ws = store.create_workspace(name=req.name, description=req.description)
    return ws

@app.get("/workspaces/{workspace_id}", response_model=WorkspaceDetail, summary="Get workspace detail")
def get_workspace(workspace_id: str):
    """Get workspace detail including conversation threads."""
    store = get_workspace_store_instance()
    ws = store.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws

@app.delete("/workspaces/{workspace_id}", summary="Delete a workspace")
def delete_workspace(workspace_id: str):
    """Delete a workspace and all cascaded conversations and messages."""
    store = get_workspace_store_instance()
    success = store.delete_workspace(workspace_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"status": "deleted", "workspace_id": workspace_id}

@app.get("/workspaces/{workspace_id}/conversations", response_model=List[ConversationSummary], summary="List conversations in a workspace")
def list_workspace_conversations(workspace_id: str):
    """List all conversation threads within a workspace."""
    store = get_workspace_store_instance()
    ws = store.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return store.list_conversations(workspace_id)

@app.post("/workspaces/{workspace_id}/conversations", response_model=ConversationDetail, summary="Create new conversation thread")
def create_workspace_conversation(workspace_id: str, req: ConversationCreate):
    """Create a new conversation thread inside a workspace."""
    store = get_workspace_store_instance()
    conv = store.create_conversation(workspace_id=workspace_id, title=req.title)
    if not conv:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return conv

@app.get("/conversations/{conversation_id}", response_model=ConversationDetail, summary="Get conversation messages")
def get_conversation(conversation_id: str):
    """Get conversation thread by ID including all message turns."""
    store = get_workspace_store_instance()
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@app.patch("/conversations/{conversation_id}", response_model=ConversationSummary, summary="Update conversation title")
def update_conversation(conversation_id: str, req: ConversationUpdate):
    """Update conversation title."""
    store = get_workspace_store_instance()
    conv = store.update_conversation_title(conversation_id, req.title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@app.delete("/conversations/{conversation_id}", summary="Delete conversation thread")
def delete_conversation(conversation_id: str):
    """Delete a conversation thread."""
    store = get_workspace_store_instance()
    success = store.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}

@app.post("/conversations/{conversation_id}/messages", response_model=MessageResponse, summary="Add message to conversation")
def add_conversation_message(conversation_id: str, req: MessagePayload):
    """Add a message turn (user or assistant) to a conversation thread."""
    store = get_workspace_store_instance()
    msg = store.add_message(
        conversation_id=conversation_id,
        role=req.role,
        content=req.content,
        sources=req.sources,
        confidence_score=req.confidence_score or 0.0,
        is_low_confidence=req.is_low_confidence or False
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return msg

@app.delete("/conversations/{conversation_id}/messages", summary="Clear conversation messages")
def clear_conversation_messages(conversation_id: str):
    """Clear all messages from a conversation thread."""
    store = get_workspace_store_instance()
    success = store.clear_conversation_messages(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "cleared", "conversation_id": conversation_id}


