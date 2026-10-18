# Research RAG System

A document search and question-answering application. It indexes documents, searches them using vector similarity and keyword matching, and generates answers with source citations using an LLM or an offline mock mode.

---

## What It Can Do

- **Ingest documents**: Upload PDFs, Word documents, PowerPoint presentations, Excel spreadsheets, text files, Markdown, Jupyter notebooks, source code, images, web pages, YouTube transcripts, and ArXiv papers.
- **Search documents**: Combines vector search (ChromaDB) and keyword matching (BM25) to find relevant passages.
- **Answer questions with citations**: Generates answers from retrieved passages with citations (`[Source 1]`, `[Source 2]`) indicating the source, section, page, or slide.
- **Filter by document**: Query all indexed documents or only selected documents using checkboxes.
- **Generate study materials**: Generate study guides, quizzes, source comparison tables, action item lists, and two-speaker audio scripts.
- **Organize with workspaces**: Group documents and conversation threads into workspaces, saved in SQLite or PostgreSQL.
- **Multiple LLM options**: Supports Google Gemini, OpenAI, Qwen, GLM, local Ollama, custom OpenAI-compatible APIs, or an offline mock mode.
- **Web and desktop interfaces**: Access via a web browser or run as a Windows desktop application.

---

## Features

### 1. Document Ingestion & Parsing
- **File formats**: Supports `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.tsv`, `.txt`, `.md`, `.ipynb`, `.html`, `.json`, `.py`, `.js`, `.ts`, `.sql`, `.png`, and `.jpg`.
- **Section extraction**: Extracts headings, slide numbers (PowerPoint), page numbers (PDF), tables, and sections such as Abstract, Introduction, Methodology, and Conclusion. Bibliography sections are excluded.
- **Web and YouTube**: Imports web pages by URL and transcripts from YouTube video links.
- **Notes**: Write or paste text notes directly into the workspace.
- **ArXiv search**: Search ArXiv by keyword or paper ID and download papers.
- **Google Drive and Google Classroom**: Connect with Google credentials to browse and import files or course materials.
- **Deduplication**: Removes duplicate documents from a workspace.

### 2. Search & Retrieval
- **ChromaDB**: Stores and searches vector embeddings locally.
- **BM25**: Keyword search using `rank-bm25`.
- **Search modes**: Hybrid (vector + BM25), vector only, or BM25 only.
- **HyDE**: Query expansion that generates a draft answer before searching.
- **Parameters**: Adjustable Top-K chunk count and minimum similarity score threshold.
- **Document catalog**: Metadata is stored in a JSON manifest; full text is loaded from disk when opened.

### 3. Question Answering & Citations
- **Citations**: Answers cite specific documents and sections.
- **Citation previews**: Citation numbers link to the retrieved passages and show text previews on hover.
- **Follow-up questions**: Generates follow-up questions after each answer.
- **Save and export**:
  - Save answers as notes in the workspace.
  - Copy answers to the clipboard.
  - Export answers to Markdown with YAML frontmatter and BibTeX references.

### 4. Study Tools
- **Audio Overview**: Generates a two-speaker script summarizing selected documents, with browser text-to-speech.
- **Study Guide**: Summarizes concepts and formulas from selected documents.
- **Practice Quiz**: Generates multiple-choice and short-answer questions with answers.
- **Compare Sources**: Generates a comparison table across selected documents.
- **Action Items**: Extracts dates, deliverables, and tasks from documents.

### 5. Workspaces
- **Workspaces**: Group documents and threads by topic.
- **Threads**: Multiple conversation threads per workspace.
- **Database**: Uses SQLite (`data/workspaces.db`) by default, or PostgreSQL if `DATABASE_URL` is set.

### 6. LLM Providers
Configured via `.env` or in the UI:
- Google Gemini
- OpenAI
- Qwen (Alibaba DashScope)
- GLM (ZhipuAI)
- Ollama (local)
- Custom OpenAI-compatible endpoints
- Offline Mock Mode (returns fixed responses without network calls or API keys)

### 7. Evaluation
- Evaluates Context Relevance, Faithfulness, and Answer Relevance.
- Uses an LLM to evaluate responses, or a token-overlap heuristic in mock mode.

---

## Installation

### Prerequisites
- Python 3.10+
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/research-rag-system.git
   cd research-rag-system
   ```

2. **Create and activate a virtual environment**:
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration (`.env`)

The default `.env` is set to mock mode:
```env
LLM_PROVIDER=mock
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

To use cloud LLM providers, add their API keys (or enter them in the UI):
```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
DASHSCOPE_API_KEY=your_qwen_api_key
ZHIPUAI_API_KEY=your_glm_api_key

# Optional: PostgreSQL connection string (defaults to SQLite if omitted)
# DATABASE_URL=postgresql://user:password@host:5432/dbname
```

---

## How to Run

### Web Interface

Start the server:
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open:
- Web UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Windows Desktop Application

Run in a native desktop window:
```powershell
python desktop_app.py
```

To build a standalone executable:
```powershell
python build_exe.py
```
Output: `dist/ResearchRAG/ResearchRAG.exe`.

### Docker

Run with Docker Compose:
```bash
docker-compose up --build
```
Access at [http://localhost:8000](http://localhost:8000).

---

## How to Use

1. **Select or create a workspace**: Use the dropdown in the header to pick or create a workspace.
2. **Add documents**: Click **Add Source** in the left panel to upload files, paste URLs or notes, search ArXiv, or connect Google Drive/Classroom.
3. **Select active sources**: Use the checkboxes in the **Sources** tab to select which documents to query.
4. **Ask questions**: Enter a question in the input box and click **Ask**.
5. **Inspect citations**: Click citation numbers to jump to retrieved chunks or hover to read the excerpt.
6. **Use study tools**: Click buttons above the query box (Audio Overview, Study Guide, Practice Quiz, Compare Sources, Action Items).
7. **Change LLM settings**: Click the model button in the header to select a provider and set API keys.
8. **Export**: Copy as Markdown, save as a workspace note, or export to an Obsidian-ready `.md` file.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` or `/ui` | Serves the web interface |
| `GET` | `/health` | Service status and chunk count |
| `POST` | `/documents/upload` | Upload and index files (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.md`, `.txt`, etc.) |
| `POST` | `/documents/ingest-url` | Ingest web page, YouTube transcript, or PDF URL |
| `POST` | `/documents/ingest-note` | Save a text note into the workspace |
| `GET` | `/documents` | List indexed documents |
| `GET` | `/documents/{doc_id}` | Retrieve document text and section outline |
| `DELETE` | `/documents/{doc_id}` | Delete a document from catalog and search indices |
| `POST` | `/documents/deduplicate` | Remove duplicate documents from the workspace |
| `POST` | `/query` | Run a RAG query with optional source filtering |
| `POST` | `/studio/generate` | Generate study guides, quizzes, comparisons, or audio scripts |
| `POST` | `/export/markdown` | Export answer to Markdown with BibTeX |
| `POST` | `/evaluate` | Run evaluation on a query and response |
| `GET` / `POST` | `/config/llm` | Get or update active LLM provider and settings |
| `POST` | `/config/llm/test` | Test LLM provider connection |
| `GET` / `POST` | `/workspaces` | List or create workspaces |
| `GET` / `POST` | `/workspaces/{id}/conversations` | List or create conversation threads |
| `POST` | `/ingest` | Download papers from ArXiv |
| `GET` / `POST` | `/gdrive/*` | Google Drive account, browse, and import |
| `GET` / `POST` | `/classroom/*` | Google Classroom courses, announcements, and import |

---

## Tests

Run tests with pytest:

```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## Project Structure

```
research-rag-system/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application and route handlers
│   │   └── schemas.py           # Request and response schemas
│   ├── indexing/
│   │   ├── document_catalog.py  # Document metadata manifest and text loading
│   │   └── vector_store.py      # ChromaDB vector store and BM25 search
│   ├── ingestion/
│   │   ├── universal_parser.py  # File, URL, and transcript parser
│   │   ├── chunker.py           # Text chunker with section header detection
│   │   ├── arxiv_collector.py   # ArXiv search client
│   │   ├── gdrive_connector.py  # Google Drive connector
│   │   └── classroom_connector.py # Google Classroom connector
│   ├── rag/
│   │   ├── pipeline.py          # Query orchestration, hybrid search, and HyDE
│   │   ├── generator.py         # LLM provider adapter
│   │   └── studio_tools.py      # Study guide, quiz, and audio overview generators
│   ├── storage/
│   │   └── workspace_store.py   # Database models for workspaces and chat history
│   ├── evals/
│   │   └── evaluator.py         # RAG evaluation metrics
│   └── config.py                # Configuration and environment settings
├── frontend/
│   └── index.html               # Web UI
├── tests/                       # Test suite
├── data/                        # Local data (ChromaDB, SQLite, document files)
├── desktop_app.py               # Desktop window runner
├── build_exe.py                 # PyInstaller build script
├── build_desktop.spec           # PyInstaller build specification
├── Dockerfile                   # Docker build file
├── docker-compose.yml           # Docker Compose configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## License

Distributed under the MIT License.