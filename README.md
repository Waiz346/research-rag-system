# 🔬 Research RAG Intelligence System

An enterprise-grade, academic-tailored Retrieval-Augmented Generation (RAG) platform and conversational research workbench—inspired by **NotebookLM** and **RAGFlow**. 

Ingest peer-reviewed papers from ArXiv, local files, or Google Drive, perform section-aware chunking, retrieve with hybrid Dense + BM25 search and HyDE query expansion, chat across selected documents with dynamic multi-provider LLM switching, benchmark answer reliability with the automated RAG Triad evaluation framework, or compile into a standalone Windows desktop executable.

---

## 🌟 Key Capabilities

- **📚 NotebookLM-Style Conversational Studio**:
  - Responsive two-column workbench: **Sources & Inspector** on the left, **Conversational Studio** on the right.
  - Multi-turn conversation preserving query history and conversational context.
  - **Active Source Scoping**: Use checkboxes to query single or multiple papers, or search across the entire library.
  - **Traceable Citations**: Expandable citation cards displaying canonical section badges (`🏷️ Methodology`, `🏷️ Abstract`), similarity scores, and verbatim excerpts.
  - **On-Demand Document Inspector**: Read full paper text, view section outlines, and manage sources without leaving the UI.

- **🗂️ Workspaces, Folders & Multi-Thread Conversations**:
  - **Hierarchical Topic Organization**: Every conversation resides in a designated Workspace / Folder (e.g. *Biomedical RAG*, *Financial Forecasting*, *NLP Survey*).
  - **Multiple Sub-Conversations per Workspace**: Switch seamlessly between parallel discussion threads within the same workspace.
  - **Automatic Thread Titling**: Conversations are automatically titled based on your initial prompt.
  - **Online Cloud Database Persistence**: Direct connection to managed cloud PostgreSQL databases (Supabase, Neon, AWS RDS) via SQLAlchemy 2.0 and pure-Python `pg8000`, with zero C compiler dependencies on Windows.
  - **Graceful Local Fallback**: Automatically falls back to local SQLite (`data/workspaces.db`) when `DATABASE_URL` is omitted, working immediately out-of-the-box.

- **📁 Google Drive Cloud Connector**:
  - Connect and browse Google Drive folders, subfolders, and files directly within the Sources workspace.
  - **5 Authentication Modes**:
    1. *Interactive Demo Mode*: Pre-loaded with canonical papers (Transformer, LoRA, RAG, Self-RAG, DPR) for instant exploration without GCP setup.
    2. *Service Account JSON*: Enterprise shared drive integration via service account key.
    3. *OAuth 2.0 Access Token*: User-scoped Bearer token authentication.
    4. *Google Cloud API Key*: Direct browsing of public/shared folders.
    5. *Local `gdrive` CLI Bridge*: Seamless integration with an existing local CLI session.
  - Visual breadcrumb navigation (`📁 My Drive / ...`) and direct folder ID/URL jumping.
  - **1-Click RAG Ingestion**: Automatically downloads PDFs/Docs, extracts sections, and indexes into ChromaDB and BM25.

- **🖥️ Standalone Windows Desktop App (`ResearchRAG.exe`)**:
  - Fully compiled native Windows application using PyInstaller and Microsoft Edge WebView2 (`pywebview`).
  - Launches with its own borderless window—no terminal windows, command lines, or browser address bars.
  - Concurrently runs the FastAPI backend and Streamlit studio in background worker threads with automated port conflict resolution.
  - Persistent data storage (`dist/ResearchRAG/data/`) located right beside the executable.

- **⚡ Memory-Efficient Document Catalog (Manifest Projection Pattern)**:
  - Lightweight disk-backed metadata projection (`manifest.json` ~200B/doc) eliminates memory overhead when listing dozens or hundreds of papers.
  - **Lazy Text Loading**: Full-text files (`data/documents/{id}.txt`) are read from disk strictly on-demand when clicked.
  - Atomic document deletion across the manifest, ChromaDB embeddings, and the BM25 index.

- **📄 Custom File Ingestion & Academic Paper Template**:
  - Drag-and-drop file upload supporting **`.pdf`**, **`.txt`**, and **`.md`** formats.
  - **Layout & Section-Aware Chunking**: Detects and normalizes canonical scientific headings (`Abstract`, `Introduction`, `Related Work`, `Methodology`, `Experiments`, `Discussion`, `Conclusion`).
  - **Bibliography Noise Pruning**: Automatically filters out reference and bibliography sections to keep citations from polluting retrieval chunks.

- **🔍 Advanced Hybrid Retrieval & Grounded Re-Ranking**:
  - **Dual-Engine Search**: Combines ChromaDB dense cosine similarity with BM25 sparse keyword ranking (featuring soft saturation scaling and stopword pruning).
  - **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical academic excerpts to bridge vocabulary gaps between natural language questions and dense scientific prose.
  - **Direct-Query Re-Scoring**: Evaluates candidate passages directly against the user's explicit question, preventing out-of-domain hallucinations.
  - **Dynamic Quality Guardrails**: Rejects irrelevant or out-of-domain queries when retrieval scores fail to clear confidence thresholds.

- **🔄 Dynamic Multi-Provider LLM Switcher**:
  - Runtime model selection with frontend API key input—no `.env` changes or server restarts required.
  - Supported providers:
    - **Google Gemini** (`gemini-2.5-flash`, `gemini-1.5-pro`)
    - **OpenAI** (`gpt-4o`, `gpt-4o-mini`)
    - **Qwen** (`qwen-plus`, `qwen-turbo` via DashScope / OpenAI compatible)
    - **GLM** (`glm-4-plus`, `glm-4-flash` via ZhipuAI)
    - **Local Ollama** (`llama3`, `mistral`, `deepseek-r1`)
    - **Offline Mock Mode** (Zero API cost, deterministic offline testing)

- **🎯 Scholarly Citation & Keyword Discovery**:
  - Ingest literature via seed citation graphs (fetching papers citing or cited by a seed paper using Semantic Scholar).
  - Narrow searches using title keywords, abstract keywords, and minimum citation thresholds.

- **📊 Automated RAG Triad Evaluation**:
  - Continuous quality scoring: **Context Relevance**, **Faithfulness (Groundedness)**, and **Answer Relevance** to guarantee factual grounding and detect hallucinations.

---

## 📐 System Architecture & Flowcharts

### 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["📥 Document Ingestion & Storage"]
        A1["ArXiv API & Semantic Scholar"] --> B["Ingestion Engine"]
        A2["Custom Uploads (.pdf, .txt, .md)"] --> B
        A3["Google Drive Connector (OAuth / SA / Demo)"] --> B
        B --> C["Academic Chunker (Paper Template)"]
        C -->|Strip References & Tag Sections| D["Chunk Collection"]
        D --> E[("ChromaDB (Dense Vectors)")]
        D --> F[("BM25 Index (Sparse Keywords)")]
        D --> G[("Manifest Projection (manifest.json)")]
        D --> H[("Raw Text Files (data/documents/*.txt)")]
    end

    subgraph RETRIEVAL["🔍 Retrieval & Reasoning Engine"]
        Q["User Question + History"] --> SCOPE{"Active Doc Filter?"}
        SCOPE -->|Yes| FILT["Scoped Chunk Subset"]
        SCOPE -->|No| ALL["All Indexed Chunks"]
        Q --> HYDE["HyDE Generator"]
        HYDE --> HYB["Hybrid Retrieval (Dense + BM25)"]
        FILT --> HYB
        ALL --> HYB
        HYB --> RESC["Direct-Query Candidate Re-Scoring"]
        RESC --> GATE{"Passes Relevance Threshold?"}
        GATE -->|No| FALLBACK["Out-of-Domain Fallback Notice"]
        GATE -->|Yes| CTX["Context Assembly with Section Badges"]
    end

    subgraph INFERENCE["🤖 Generation & Evaluation"]
        CTX --> LLM["Dynamic LLM Interface<br/>(Gemini / OpenAI / Qwen / GLM / Ollama / Mock)"]
        LLM --> RESP["Cited Response with Section Attribution"]
        RESP --> EVAL["RAG Triad Evaluator<br/>(Context Rel / Faithfulness / Answer Rel)"]
    end

    subgraph CLIENT["💻 Client Layer"]
        UI["Streamlit Conversational Studio (NotebookLM UI)"] <--> FASTAPI["FastAPI REST Service (port 8000)"]
        DESK["Standalone Desktop Shell (ResearchRAG.exe / Edge WebView2)"] <--> UI
        FASTAPI <--> RETRIEVAL
        FASTAPI <--> INGESTION
        FASTAPI <--> INFERENCE
    end
```

---

### 2. Retrieval, Reasoning & Citation Pipeline

```mermaid
flowchart LR
    subgraph INPUT["1. Input & Scope"]
        U["User Query"]
        H["Chat History"]
        S["Active Source IDs"]
    end

    subgraph RETRIEVE["2. Hybrid Multi-Stage Retrieval"]
        U --> HYDE["Generate Hypothetical<br/>Academic Passage (HyDE)"]
        HYDE --> DENSE["ChromaDB Cosine<br/>Dense Search"]
        U --> BM25["BM25 Lexical<br/>Sparse Search"]
        DENSE --> RRF["Score Normalization<br/>& Reciprocal Rank Fusion"]
        BM25 --> RRF
        S -.->|Filter by doc_id| DENSE
        S -.->|Filter by doc_id| BM25
    end

    subgraph RESCORE["3. Grounded Verification"]
        RRF --> SCR["Candidate Re-Scorer<br/>(Grounded in Direct Query)"]
        SCR --> THRESH{"Score >= 0.50?"}
        THRESH -->|No| REJ["Reject with Helpful Guide"]
        THRESH -->|Yes| PASS["Top-K Relevant Chunks"]
    end

    subgraph SYNTHESIS["4. Synthesis & Audit"]
        PASS --> PROMPT["Context Assembly<br/>[Source N] [Section]"]
        PROMPT --> GEN["LLM Generation"]
        GEN --> ANS["Synthesized Answer"]
        ANS --> TRIAD["RAG Triad Benchmark<br/>(Faithfulness & Relevance)"]
    end
```

---

### 3. Google Drive Connector & Ingestion Flow

```mermaid
flowchart TD
    subgraph GDRIVE["☁️ Google Drive"]
        F1["📁 Folders & Subfolders"]
        P1["📄 Research PDFs"]
        D1["📝 Google Docs & Markdown"]
    end

    subgraph CONNECTOR["🔌 GoogleDriveConnector (src/ingestion/gdrive_connector.py)"]
        AUTH{"Authentication & Account Selection"}
        AUTH -->|1-Click Fast Switch| SW["Multi-Account Store (data/gdrive_accounts/)"]
        AUTH -->|OAuth Chooser| OAC["Browser Sign-in (prompt='select_account')"]
        AUTH -->|Enterprise| SA["Service Account JSON"]
        AUTH -->|Manual Token| OA["OAuth 2.0 Access Token"]
        AUTH -->|Public Links| AK["Google API Key"]
        AUTH -->|Instant Test| DEMO["Interactive Demo Mode"]
    end

    subgraph ACTIONS["🔄 Ingestion Actions"]
        BROWSE["Folder Tree & Breadcrumbs Browser"]
        DOWN["Binary & Doc Downloader (pypdf/export)"]
        CHUNK["Academic Section-Aware Chunker"]
        INDEX[("ChromaDB + BM25 + Manifest Index")]
    end

    GDRIVE <--> CONNECTOR
    CONNECTOR --> BROWSE
    BROWSE -->|User clicks 'Import into RAG'| DOWN
    DOWN --> CHUNK --> INDEX
```

---

### 4. Memory-Efficient Document Lifecycle & Inspector

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Researcher
    participant UI as Streamlit Web UI
    participant API as FastAPI Backend
    participant Cat as Document Catalog
    participant Disk as Local Disk Storage
    participant Vec as ChromaDB & BM25

    Note over User, Vec: Ingestion Phase
    User->>UI: Uploads paper (or imports from Google Drive / ArXiv)
    UI->>API: POST /documents/upload (or /gdrive/import, /ingest)
    API->>API: Section-aware chunking & reference pruning
    API->>Vec: Add embeddings & tokens
    API->>Disk: Save raw text to data/documents/{id}.txt
    API->>Cat: Append metadata to manifest.json (~200 bytes)
    API-->>UI: Ingestion succeeded

    Note over User, Vec: Memory-Efficient Browsing Phase
    User->>UI: Opens Sources Panel
    UI->>API: GET /documents
    API->>Cat: Read manifest.json (Fast, <2MB RAM)
    API-->>UI: Returns metadata summaries
    UI-->>User: Displays source cards with checkboxes

    Note over User, Vec: On-Demand Lazy Reading
    User->>UI: Clicks "View & Read" on Paper #1
    UI->>API: GET /documents/{doc_id}
    API->>Disk: Lazily read data/documents/{doc_id}.txt
    API-->>UI: Full document content + section breakdown
    UI-->>User: Render paper in Inspector
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11 & 3.13)
- **Git**

### 2. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/your-username/research-rag-system.git
cd research-rag-system

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell
# source .venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
The repository includes a default `.env` configured for **Offline Mock Mode**, allowing instant testing without API keys:
```env
LLM_PROVIDER=mock
CHROMA_PERSIST_DIRECTORY=./data/chroma
DOCUMENT_DIR=./data/documents
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K=4
```

To configure default API keys for cloud providers (optional—keys can also be provided directly in the UI):
```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
DASHSCOPE_API_KEY=your_qwen_api_key
ZHIPUAI_API_KEY=your_glm_api_key
```

### 4. Online Cloud Database Configuration (Workspaces & Conversations)
By default, the system stores workspaces, sub-conversations, and message history locally in SQLite (`data/workspaces.db`). To synchronize and persist across team environments or host in the cloud, configure `DATABASE_URL` in `.env`:

```env
# Example 1: Supabase (PostgreSQL with pure-Python driver)
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres

# Example 2: Neon Serverless Postgres
DATABASE_URL=postgresql://[USER]:[PASSWORD]@[ENDPOINT].neon.tech/neondb?sslmode=require

# Example 3: Local SQLite fallback (default when DATABASE_URL is unset)
# DATABASE_URL=sqlite:///data/workspaces.db
```
*Note: Any standard `postgres://` or `postgresql://` URI is automatically adapted to pure-Python `postgresql+pg8000://` with zero external C-compiler dependencies.*

---

## 🏃 Running the Application

### Option A: Local Execution (Recommended)

1. **Start the FastAPI Backend Service**:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   - Interactive OpenAPI Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

2. **Start the Streamlit Conversational Studio**:
   ```powershell
   .\.venv\Scripts\streamlit.exe run frontend/app.py --server.port=8501
   ```
   - Web Application: [http://localhost:8501](http://localhost:8501)

---

### Option B: Standalone Desktop Application (`ResearchRAG.exe`)

You can launch the system as a self-contained Windows desktop app with its own native window:

1. **Direct Launch**:
   - Navigate to `dist/ResearchRAG/`
   - Double-click **`ResearchRAG.exe`** (or `Launch_Research_RAG.bat`)
   - It boots both the backend API and Streamlit studio automatically and opens a native desktop window powered by Microsoft Edge WebView2.

2. **Re-compiling the Executable**:
   If you modify code and want to recompile the desktop application:
   ```powershell
   .\.venv\Scripts\python.exe build_exe.py
   ```
   This automates the PyInstaller compilation using `build_desktop.spec` and updates the standalone bundle inside `dist/ResearchRAG/`.

---

### Option C: Docker Compose

Launch the full stack in isolated containers:
```bash
docker-compose up --build
```
- **FastAPI**: `http://localhost:8000`
- **Streamlit UI**: `http://localhost:8501`

---

## 🧭 How to Use the NotebookLM-Style Studio

1. **Add Sources**:
   - Navigate to the **📚 Sources** column on the left.
   - **Upload Tab**: Drag and drop `.pdf`, `.txt`, or `.md` papers.
   - **ArXiv Tab**: Search for papers by keyword or citation network (e.g., papers citing `"2005.11401"`).
   - **Google Drive Tab**: Connect via Service Account, OAuth token, API key, local `gdrive` CLI, or Demo mode. Browse folders and click **"📥 Import into RAG"** to ingest Drive papers directly.
2. **Select Active Scope**:
   - Check or uncheck papers in the source list. 
   - Questions will be answered **strictly** using checked papers, or across the whole library if none are selected.
3. **Inspect Documents**:
   - Click **"View & Read"** on any paper to open the on-demand inspector and examine its full text and detected academic sections.
4. **Chat in the Conversational Studio**:
   - Ask research questions in the right column.
   - Inspect the expandable citations beneath each answer to view source section badges (`🏷️ Methodology`, `🏷️ Experiments`), similarity scores, and exact excerpts.
5. **Switch Models Dynamically**:
   - Use the sidebar model selector to switch between Gemini, OpenAI, Qwen, GLM, Ollama, and Mock Mode on the fly.

---

## 🔌 API Reference Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/documents/upload` | Multipart file upload (`.pdf`, `.txt`, `.md`) with section chunking and indexing |
| `GET` | `/documents` | Memory-efficient list of indexed documents (~200B metadata manifest per doc) |
| `GET` | `/documents/{doc_id}` | On-demand lazy retrieval of full document text and section outline |
| `DELETE` | `/documents/{doc_id}` | Cascading deletion of a document from manifest, ChromaDB, and BM25 index |
| `POST` | `/ingest` | Automated ArXiv fetch by citation seed, keywords, and minimum citations |
| `POST` | `/query` | Multi-turn RAG query with source filtering (`doc_ids`), history, and dynamic LLM switcher |
| `GET` | `/gdrive/status` | Check Google Drive connector status and active auth mode |
| `POST` | `/gdrive/connect` | Authenticate Google Drive via Service Account, OAuth token, API key, gdrive CLI, or Demo |
| `GET` | `/gdrive/browse` | Browse Google Drive folders and list subfolders/files with breadcrumbs |
| `POST` | `/gdrive/import` | Download and index a Google Drive file directly into ChromaDB + BM25 |
| `GET` | `/health` | Service health status, active index chunk count, and catalog count |

---

## 🧪 Testing & Quality Assurance

Run the complete test suite:
```powershell
.\.venv\Scripts\pytest tests/ -v
```

The test suite covers:
- `test_chunker.py`: Canonical section detection, fallback sliding windows, and reference pruning.
- `test_hybrid_search.py`: BM25 tokenization, dense cosine search, and reciprocal rank fusion.
- `test_arxiv_collector.py`: Citation enrichment and query parameter formatting.
- `test_api.py`: Multipart file uploads, document catalog endpoints, lazy text inspection, multi-turn history handling, and source-scoped retrieval.
- `test_gdrive.py`: Google Drive connector authentication modes, folder hierarchy browsing, file text extraction, and direct RAG indexing.

---

## 📂 Project Structure

```
research-rag-system/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI endpoints, upload handler, GDrive & catalog routing
│   │   └── schemas.py           # Pydantic v2 data models & request/response schemas
│   ├── indexing/
│   │   ├── document_catalog.py  # Manifest projection pattern & lazy text loader
│   │   └── vector_store.py      # ChromaDB dense + BM25 sparse hybrid retrieval engine
│   ├── ingestion/
│   │   ├── arxiv_collector.py   # ArXiv search, Semantic Scholar citation enrichment
│   │   ├── chunker.py           # Academic paper template chunker & section normalizer
│   │   └── gdrive_connector.py  # Google Drive multi-mode auth, folder browsing & ingestion
│   ├── rag/
│   │   ├── generator.py         # Multi-provider LLM adapter (Gemini, OpenAI, Qwen, GLM, Ollama, Mock)
│   │   └── pipeline.py          # HyDE expansion, candidate re-scoring, & history synthesis
│   ├── evals/
│   │   └── evaluator.py         # Automated RAG Triad benchmarking (Faithfulness, Relevance)
│   └── config.py                # Centralized environment settings
├── frontend/
│   └── app.py                   # Streamlit NotebookLM-style two-column workbench
├── tests/                       # Pytest unit & integration test suite (17 tests)
│   ├── test_api.py
│   ├── test_arxiv_collector.py
│   ├── test_chunker.py
│   ├── test_gdrive.py
│   └── test_hybrid_search.py
├── data/
│   ├── chroma_db/               # Persistent ChromaDB vector files
│   ├── documents/               # Disk-backed full text files (lazy loading)
│   └── manifest.json            # Lightweight metadata manifest
├── desktop_app.py               # Native desktop window runner (PyWebView + Edge WebView2)
├── build_desktop.spec           # PyInstaller build specification
├── build_exe.py                 # Automated compilation script for ResearchRAG.exe
├── Dockerfile                   # Container build definition
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## 📜 License & Citation

Distributed under the MIT License. If you use this software in your research or workflows, feel free to cite or star the repository.