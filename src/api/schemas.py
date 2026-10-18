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
    workspace_id: Optional[str] = Field(default="default", description="Workspace/Folder to store fetched papers in")

class PaperSummaryInfo(BaseModel):
    id: str
    title: str
    citation_count: int = 0

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message author role: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

class DocumentSummarySchema(BaseModel):
    doc_id: str
    workspace_id: Optional[str] = "default"
    title: str
    source_type: str
    filename: str
    authors: str = "Unknown"
    citation_count: int = 0
    chunk_count: int = 0
    sections: List[str] = []
    created_at: str = ""
    char_count: int = 0
    summary: str = ""
    key_topics: List[str] = []
    suggested_questions: List[str] = []

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
    workspace_id: Optional[str] = Field(default=None, description="Scope query strictly to documents within this workspace")
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
    rewritten_query: Optional[str] = None
    query_intent: Optional[str] = "conceptual_research"
    follow_up_questions: List[str] = []

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
    database_engine: Optional[str] = "SQLite (Local Fallback)"

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

# ==============================================================================
# Google Drive Connector Schemas
# ==============================================================================

class GDriveConnectRequest(BaseModel):
    auth_type: str = Field(default="demo", description="demo, api_key, access_token, service_account, or gdrive_cli")
    api_key: Optional[str] = Field(default=None, description="Google Cloud API Key")
    access_token: Optional[str] = Field(default=None, description="OAuth 2.0 Bearer Access Token")
    service_account_json: Optional[str] = Field(default=None, description="Service Account JSON string or file path")

class GDriveStatusResponse(BaseModel):
    connected: bool
    auth_type: str
    account_email: Optional[str] = None
    has_detected_account: bool = False
    detected_email: Optional[str] = None
    message: str

class GDriveItem(BaseModel):
    id: str
    name: str
    mime_type: str
    is_folder: bool
    size: Optional[int] = None
    modified_time: Optional[str] = None
    web_view_link: Optional[str] = None
    icon: Optional[str] = "📄"

class GDriveBrowseResponse(BaseModel):
    current_folder_id: str
    current_folder_name: str
    parent_id: Optional[str] = None
    folders: List[GDriveItem] = []
    files: List[GDriveItem] = []
    total_items: int

class GDriveSearchResponse(BaseModel):
    query: str
    folders: List[GDriveItem] = []
    files: List[GDriveItem] = []
    total_items: int

class GDriveImportRequest(BaseModel):
    file_id: str
    title: Optional[str] = Field(default=None, description="Optional override title")
    workspace_id: Optional[str] = Field(default="default", description="Workspace/Folder to store document under")
    section_aware: bool = Field(default=True, description="Enable academic section chunking")
    exclude_references: bool = Field(default=True, description="Strip bibliography")

class GDriveImportResponse(BaseModel):
    status: str
    doc_id: str
    title: str
    filename: str
    chunks_indexed: int
    total_in_index: int
    sections: List[str]
    char_count: int

class GDriveAccountInfo(BaseModel):
    email: str
    name: Optional[str] = None
    is_active: bool = False
    updated_at: Optional[str] = None

class GDriveAccountsResponse(BaseModel):
    accounts: List[GDriveAccountInfo] = []
    active_account: Optional[str] = None
    count: int = 0

class GDriveSwitchAccountRequest(BaseModel):
    email: str

class GDriveLoginStatusResponse(BaseModel):
    status: str  # "idle", "in_progress", "completed", "error"
    auth_url: Optional[str] = None
    email: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class GDriveBatchImportItem(BaseModel):
    file_id: str
    title: Optional[str] = None

class GDriveBatchImportRequest(BaseModel):
    items: List[GDriveBatchImportItem] = []
    workspace_id: Optional[str] = Field(default="default", description="Workspace/Folder to store documents under")
    section_aware: bool = Field(default=True)
    exclude_references: bool = Field(default=True)

class GDriveBatchImportResponse(BaseModel):
    total_requested: int
    total_succeeded: int
    total_failed: int
    imported_docs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

class BatchDeleteRequest(BaseModel):
    doc_ids: List[str]

class BatchDeleteResponse(BaseModel):
    deleted_count: int
    deleted_ids: List[str]


# ==============================================================================
# Workspace & Multi-Conversation Schemas
# ==============================================================================

class MessagePayload(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Cited sources breakdown")
    confidence_score: Optional[float] = Field(default=0.0, description="RAG confidence score")
    is_low_confidence: Optional[bool] = Field(default=False, description="Flag for low confidence warning")

class MessageResponse(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    sources: List[Dict[str, Any]] = []
    confidence_score: float = 0.0
    is_low_confidence: bool = False
    created_at: str

class ConversationSummary(BaseModel):
    id: str
    workspace_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

class ConversationDetail(BaseModel):
    id: str
    workspace_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    messages: List[MessageResponse] = []

class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default="New Conversation", description="Title of the conversation thread")

class ConversationUpdate(BaseModel):
    title: str = Field(..., description="Updated title for conversation thread")

class WorkspaceSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    created_at: str
    updated_at: str
    conversation_count: int = 0
    document_count: int = 0
    conversations: List[ConversationSummary] = []

class WorkspaceDetail(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    created_at: str
    updated_at: str
    conversation_count: int = 0
    document_count: int = 0
    conversations: List[ConversationSummary] = []

class WorkspaceCreate(BaseModel):
    name: str = Field(..., description="Name of the workspace or folder")
    description: Optional[str] = Field(default="", description="Optional workspace description")

class WorkspaceListResponse(BaseModel):
    workspaces: List[WorkspaceSummary] = []
    active_workspace_id: Optional[str] = None
    count: int = 0


# ==============================================================================
# Google Classroom Connector Schemas
# ==============================================================================

class ClassroomStatusResponse(BaseModel):
    connected: bool
    auth_type: str
    account_email: Optional[str] = None
    courses_count: int = 0
    message: str
    latency_ms: Optional[float] = None
    valid: bool = True
    error: Optional[str] = None

class ClassroomConnectRequest(BaseModel):
    auth_type: str = Field(default="access_token", description="'user_account', 'service_account', or 'access_token'")
    access_token: Optional[str] = None
    service_account_json: Optional[str] = None

class ClassroomOAuthConfigRequest(BaseModel):
    client_id: str = Field(..., description="Google Cloud OAuth Client ID")
    client_secret: str = Field(..., description="Google Cloud OAuth Client Secret")


class ClassroomCourseItem(BaseModel):
    id: str
    name: str
    section: Optional[str] = ""
    description_heading: Optional[str] = ""
    room: Optional[str] = ""
    course_state: str = "ACTIVE"
    alternate_link: Optional[str] = ""
    enrollment_code: Optional[str] = ""

class ClassroomCoursesResponse(BaseModel):
    courses: List[ClassroomCourseItem] = []
    total_courses: int = 0

class ClassroomMaterialItem(BaseModel):
    id: str
    title: str
    type: str  # "drive_file", "link", "youtube", "form"
    drive_file_id: Optional[str] = None
    url: Optional[str] = ""
    mime_type: Optional[str] = None
    thumbnail_url: Optional[str] = None

class ClassroomAnnouncementItem(BaseModel):
    id: str
    course_id: Optional[str] = ""
    text: str
    created_at: Optional[str] = ""
    alternate_link: Optional[str] = ""
    materials: List[ClassroomMaterialItem] = []

class ClassroomCourseworkMaterialItem(BaseModel):
    id: str
    course_id: Optional[str] = ""
    title: str
    description: Optional[str] = ""
    created_at: Optional[str] = ""
    alternate_link: Optional[str] = ""
    materials: List[ClassroomMaterialItem] = []

class ClassroomCourseworkItem(BaseModel):
    id: str
    course_id: Optional[str] = ""
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = None
    max_points: Optional[float] = None
    alternate_link: Optional[str] = ""
    materials: List[ClassroomMaterialItem] = []

class ClassroomCourseStreamResponse(BaseModel):
    course_id: str
    course_name: str
    announcements: List[ClassroomAnnouncementItem] = []
    coursework_materials: List[ClassroomCourseworkMaterialItem] = []
    coursework: List[ClassroomCourseworkItem] = []
    total_items: int = 0

class ClassroomImportItem(BaseModel):
    item_type: str = Field(default="drive_file", description="'drive_file', 'announcement_text', or 'material_text'")
    course_id: str
    course_name: Optional[str] = None
    drive_file_id: Optional[str] = None
    title: str
    mime_type: Optional[str] = None
    raw_text: Optional[str] = None

class ClassroomImportRequest(BaseModel):
    course_id: str
    course_name: Optional[str] = None
    item_type: str = Field(default="drive_file")
    drive_file_id: Optional[str] = None
    title: str
    mime_type: Optional[str] = None
    raw_text: Optional[str] = None
    workspace_id: Optional[str] = Field(default="default", description="Workspace to store document under")
    section_aware: bool = Field(default=True)
    exclude_references: bool = Field(default=True)

class ClassroomImportResponse(BaseModel):
    status: str
    doc_id: str
    title: str
    course_name: str
    chunks_indexed: int
    total_in_index: int
    sections: List[str]
    char_count: int

class ClassroomBatchImportRequest(BaseModel):
    items: List[ClassroomImportItem] = []
    workspace_id: Optional[str] = Field(default="default")
    section_aware: bool = Field(default=True)
    exclude_references: bool = Field(default=True)

class ClassroomBatchImportResponse(BaseModel):
    total_requested: int
    total_succeeded: int
    total_failed: int
    imported_docs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

class ClassroomAccountInfo(BaseModel):
    email: str
    name: Optional[str] = None
    is_active: bool = False
    has_classroom_scopes: bool = True
    updated_at: Optional[str] = None

class ClassroomAccountsResponse(BaseModel):
    accounts: List[ClassroomAccountInfo] = []
    active_account: Optional[str] = None
    count: int = 0

class ClassroomSwitchAccountRequest(BaseModel):
    email: str

class ClassroomLoginStatusResponse(BaseModel):
    status: str
    auth_url: Optional[str] = None
    email: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


# ==============================================================================
# Everyday Ingestion, Deduplication, Studio Tools & Export Schemas
# ==============================================================================

class UrlIngestRequest(BaseModel):
    url: str = Field(..., description="Web article URL, YouTube video link, or direct PDF URL")
    title: Optional[str] = Field(default=None, description="Optional custom title override")
    workspace_id: Optional[str] = Field(default="default", description="Target workspace ID")
    section_aware: bool = Field(default=True, description="Enable section-aware chunking")
    exclude_references: bool = Field(default=True, description="Exclude reference list noise")

class NoteIngestRequest(BaseModel):
    title: str = Field(..., description="Title of the quick note, lecture summary, or syllabus paste")
    content: str = Field(..., description="Raw text or Markdown content to index")
    workspace_id: Optional[str] = Field(default="default", description="Target workspace ID")
    section_aware: bool = Field(default=True, description="Enable section-aware chunking")

class DeduplicateResponse(BaseModel):
    status: str = "success"
    duplicates_removed: int = 0
    removed_doc_ids: List[str] = []
    remaining_documents: int = 0

class StudioGenerateRequest(BaseModel):
    tool_type: Optional[str] = Field(
        default=None,
        description="'study_guide', 'quiz', 'comparison_matrix', 'action_items', or 'executive_briefing'"
    )
    action_type: Optional[str] = Field(
        default=None,
        description="Alias for tool_type"
    )
    workspace_id: Optional[str] = Field(default="default", description="Workspace scope")
    doc_ids: Optional[List[str]] = Field(default=None, description="Specific document IDs to include")
    custom_focus: Optional[str] = Field(default=None, description="Optional topic or exam focus")

class StudioGenerateResponse(BaseModel):
    tool_type: str
    action_type: str
    title: str
    content: str
    doc_count: int = 0
    sources: List[CitationSchema] = []
    latency_ms: float

class MarkdownExportRequest(BaseModel):
    title: str = Field(default="NexusRAG Research Synthesis")
    query: str = Field(default="")
    content: Optional[str] = Field(default=None)
    answer: Optional[str] = Field(default=None)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    workspace_id: str = Field(default="default")
    workspace_name: Optional[str] = Field(default=None)
    include_bibtex: bool = Field(default=True)
    obsidian_compatible: bool = Field(default=True)

class MarkdownExportResponse(BaseModel):
    markdown: str
    bibtex: str = ""
    filename: str

class NotebookGuideResponse(BaseModel):
    workspace_id: str = "default"
    source_count: int = 0
    overview: str = ""
    key_topics: List[str] = []
    suggested_questions: List[str] = []

