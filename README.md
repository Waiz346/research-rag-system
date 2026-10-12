# Research RAG Intelligence System

An end-to-end GenAI & Machine Learning integration system that ingests scientific literature from ArXiv, constructs a persistent semantic vector index, performs citation-grounded retrieval, serves inference via a FastAPI REST service, and benchmarks quality using the automated RAG Triad evaluation framework.

---

## Features

- **Automated Data Ingestion**: Programmatic ArXiv scraper fetching research metadata, abstracts, and optional PDF text extraction.
- **Configurable Document Chunking**: Sliding-window chunker with dynamic boundary snapping, configurable overlaps, and complete provenance/metadata retention.
- **Embedded Vector Database**: Persistent ChromaDB indexing enabling cosine similarity search with normalized distance metrics.
- **Provider-Agnostic LLM Interface**: Seamless switching between **Offline Mock Mode** (zero cost/API keys needed), **OpenAI** (`gpt-4o-mini`), **Google Gemini** (`gemini-1.5-flash`), or **Local Ollama** (`llama3`).
- **Production REST API**: FastAPI backend with Pydantic v2 schemas, CORS middleware, structured error handling, and Swagger UI documentation.
- **Automated Evaluation Suite**: Automated implementation of the RAG Triad:
  - **Context Relevance**: Measures whether retrieved chunks match query concepts.
  - **Faithfulness (Groundedness)**: Verifies that statements in the answer are supported by retrieved excerpts to prevent hallucinations.
  - **Answer Relevance**: Checks that the synthesized output addresses the user's intent.
- **Interactive UI**: Streamlit visual dashboard with literature explorer, citation cards, and real-time evaluation gauges.
- **Containerized**: `Dockerfile` and multi-service `docker-compose.yml`.

---

## 📐 System Architecture

```
[ ArXiv API / PDFs ]
         │
         ▼
[ Ingestion & Sliding-Window Chunker (src/ingestion/) ]
         │
         ▼
[ ChromaDB Vector Store (src/indexing/) ]
         │
    ┌────┴───────────────────────────────┐
    │                                    │
    ▼                                    ▼
[ Semantic Retrieval ]         [ Automated Evals (src/evals/) ]
    │                                    │
    ▼                                    ▼
[ LLM Synthesis + Citations ]   [ Context Relevance / Faithfulness / Answer Relevance ]
    │
    ▼
[ FastAPI Service (src/api/) ] ◄───► [ Streamlit UI (frontend/) ]
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Tested with Python 3.11 / 3.13)
- Git

### 2. Setup Virtual Environment
Open PowerShell or Terminal inside this directory:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (For Command Prompt: .venv\Scripts\activate.bat)
# (For Mac/Linux: source .venv/bin/activate)
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configuration
A default `.env` file is included with `LLM_PROVIDER=mock`, allowing test without API key

If you want to use OpenAI or Gemini:
1. Open `.env`
2. Change `LLM_PROVIDER=openai` (or `gemini`)
3. Add your `OPENAI_API_KEY=...` or `GEMINI_API_KEY=...`

---

## Running the System

### Option A: Local Run 

**Step 1: Start the FastAPI Backend**
```powershell
uvicorn src.api.main:app --reload --port 8000
```
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

**Step 2: Start the Streamlit Dashboard (In a separate terminal)**
```powershell
streamlit run frontend/app.py
```
- Frontend UI: [http://localhost:8501](http://localhost:8501)

---

### Option B: Docker Compose
```bash
docker-compose up --build
```
- API runs on `http://localhost:8000`
- Web Dashboard runs on `http://localhost:8501`

---

## 🧪 Running Automated Tests
```powershell
pytest tests/ -v
```

---

## 📂 Project Organization

```
research-rag-system/
├── src/
│   ├── api/             # FastAPI app, schemas, routing
│   ├── ingestion/       # ArXiv collector, PDF parser, chunker
│   ├── indexing/        # ChromaDB client & vector operations
│   ├── rag/             # Generator (Mock/OpenAI/Gemini/Ollama) & pipeline
│   ├── evals/           # Evaluation metrics & scoring algorithms
│   └── config.py        # Centralized Pydantic settings
├── frontend/
│   └── app.py           # Streamlit graphical interface
├── tests/               # Pytest unit and integration test suite
├── data/                # Local database and downloaded papers
├── Dockerfile           # Production container build
├── docker-compose.yml   # Multi-container orchestration
├── requirements.txt     # Locked dependencies
├── STEP_BY_STEP_GUIDE.md# Educational deep-dive roadmap
└── README.md            # Project documentation
```