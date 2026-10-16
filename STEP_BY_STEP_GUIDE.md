# 🎓 End-to-End GenAI & RAG System: Complete Step-by-Step Guide

This guide is designed for Data Science students looking to master production Machine Learning / GenAI systems. It explains every architectural phase of this project, why specific engineering decisions were made, and how to expand each module.

---

## Table of Contents
1. [The Mindset Shift: From Notebooks to Production Systems](#chapter-1-the-mindset-shift)
2. [Phase 1: Ingestion & Smart Chunking](#chapter-2-phase-1-ingestion--smart-chunking)
3. [Phase 2: Vector Embeddings & Persistent Storage](#chapter-3-phase-2-vector-embeddings--persistent-storage)
4. [Phase 3: Retrieval, Grounding & Citations](#chapter-4-phase-3-retrieval-grounding--citations)
5. [Phase 4: Serving Inference with FastAPI](#chapter-5-phase-4-serving-inference-with-fastapi)
6. [Phase 5: Automated Evaluations (The RAG Triad)](#chapter-6-phase-5-automated-evaluations)
7. [Phase 6: Interactive Web Workbench (`frontend/index.html`)](#chapter-7-phase-6-interactive-dashboard)
8. [Phase 7: Containerization & Deployment](#chapter-8-phase-7-containerization--deployment)
9. [Portfolio & Interview Presentation Guide](#chapter-9-portfolio--interview-guide)

---

## Chapter 1: The Mindset Shift

In university courses and Kaggle competitions, data science workflows often look like this:
```
Download CSV ──► Open Jupyter Notebook ──► model.fit() ──► Plot Confusion Matrix
```

In industry, **less than 20% of an AI engineer's time is spent tuning models**. The rest is spent on:
- Ensuring data pipelines don't fail when upstream schemas change.
- Handling latency, rate limits, and concurrent user requests.
- Validating whether the model hallucinates or drifts over time.
- Writing unit tests and packaging services in Docker containers.

This project implements that entire lifecycle.

---

## Chapter 2: Phase 1: Ingestion & Smart Chunking

### The Problem
Large Language Models have finite context windows. Feeding an entire 40-page research paper into an LLM for every user query is slow, expensive, and dilutes the model's attention.

### The Implementation (`src/ingestion/`)
1. **Collector (`arxiv_collector.py`)**:
   - Queries the public ArXiv API for papers matching specific topics.
   - Parses author names, dates, paper abstracts, and optionally downloads PDFs via `pypdf`.
2. **Chunker (`chunker.py`)**:
   - Instead of naive character slicing (which cuts words in half), our chunker uses a **sliding window** with `chunk_size=800` and `chunk_overlap=150`.
   - It snaps boundaries to the end of full sentences (`. `).
   - **Crucial Step**: It attaches metadata (paper title, author, ArXiv ID, chunk index) to every single chunk so they can be traced back.

### Exercise to Extend:
- Add support for Markdown files or HTML web scraping using `BeautifulSoup`.

---

## Chapter 3: Phase 2: Vector Embeddings & Persistent Storage

### The Mechanics (`src/indexing/vector_store.py`)
- We use **ChromaDB**, an embedded vector database that runs directly on your disk without needing an external cluster or cloud subscription.
- When chunks are upserted:
  1. The text is passed to an embedding model (by default, `all-MiniLM-L6-v2`).
  2. The model outputs a high-dimensional dense vector representing the semantics.
  3. The vector is saved alongside its raw text and metadata.
- When searching, the user query is also embedded into a vector, and Chroma computes the **cosine distance** between the query and stored chunks.

---

## Chapter 4: Phase 3: Retrieval, Grounding & Citations

### Preventing Hallucinations (`src/rag/`)
A naive RAG prompt says: *"Answer the question using the context."*
A production prompt enforces **strict grounding and provenance**:

```text
Context from research papers:
[Source 1] Title: ...
Content: ...

[Source 2] Title: ...
Content: ...

User Question: {question}

Provide an accurate answer synthesized strictly from the above sources.
Always cite the source numbers (e.g. [Source 1], [Source 2]).
If the context does not contain enough information, state that clearly.
```

The system packages the response into a structured `RAGResponse` containing:
- Synthesized text.
- Formatted list of `SourceCitation` objects (paper title, score, snippet, link).
- Millisecond-precision latency measurement.

---

## Chapter 5: Phase 4: Serving Inference with FastAPI

### Why FastAPI? (`src/api/`)
Data science models belong behind standard HTTP REST APIs.
- FastAPI provides automatic request validation using **Pydantic v2**. If a client passes an invalid query or missing parameters, FastAPI automatically rejects it with clean error codes.
- It automatically generates interactive Swagger documentation at `/docs`.
- It uses Python's `asyncio` for high-throughput, non-blocking I/O.

### Key Endpoints:
- `GET /`: Serves the standalone NexusRAG HTML5/Tailwind/ES6 Research Workbench.
- `POST /ingest`: Triggers paper collection and vector indexing.
- `POST /query`: Semantic retrieval and LLM answer generation.
- `POST /evaluate`: Runs the evaluation pipeline on demand.
- `GET /health`: Returns service health and indexed document counts.

---

## Chapter 6: Phase 5: Automated Evaluations (The RAG Triad)

### How do we know if our system is working? (`src/evals/evaluator.py`)
In traditional ML, you compute accuracy, precision, and F1 score against a test set. In generative AI, answers are free-form text.

We implement the **RAG Triad**:

1. **Context Relevance**:
   - Did our retriever pull back chunks that actually relate to what the user asked?
   - Measured by analyzing concept overlap between query tokens and retrieved snippets.
2. **Faithfulness (Groundedness)**:
   - Did the LLM fabricate facts, or are all statements grounded in the retrieved sources?
   - Prevents hallucinations.
3. **Answer Relevance**:
   - Does the generated answer directly address the user's question, rather than going on an irrelevant tangent?
4. **Composite Score**:
   - Weighted harmonic combination providing a single 0–100 benchmark score.

---

## Chapter 7: Phase 6: Interactive Dashboard

### Standalone NexusRAG Workbench (`frontend/index.html`)
Rather than relying on heavy server-side widget frameworks, we serve a responsive, zero-dependency **HTML5 + Tailwind CSS + ES6 JavaScript** single-page application directly from FastAPI:
- **Search & Ingest Workbench**: Upload PDFs, query ArXiv, or connect Google Drive & Classroom with a single click.
- **Synthesized Research Report**: View grounded answers with interactive inline citation pills (`[1]`, `[2]`) that auto-scroll to the underlying vector chunk.
- **Vector Chunk Inspector & RAG Triad**: Real-time similarity cutoff filtering, keyword highlighting, and automated Context Relevance, Faithfulness, and Answer Relevance bars.

---

## Chapter 8: Phase 7: Containerization & Deployment

### Docker (`Dockerfile` & `docker-compose.yml`)
*"It works on my machine"* is unacceptable in production engineering.
- The `Dockerfile` packages the Python environment, dependencies, and standalone UI into a self-contained image.
- `docker-compose.yml` launches the unified FastAPI backend + NexusRAG Web Workbench on port `8000`, mapping persistent storage volumes to `./data`.

---

## Chapter 9: Portfolio & Interview Presentation Guide

### 1. GitHub Repository Checklist
- Push this code to GitHub with a descriptive title: `arxiv-rag-system` or `production-rag-eval-pipeline`.
- Include screenshots of the NexusRAG Workbench in your `README.md`.
- Add a badge for Pytest passing.

### 2. Bullet Points for Your Resume
```markdown
- Built an end-to-end ArXiv research intelligence pipeline using FastAPI, ChromaDB, and Python, indexing 1,000+ chunks with custom sliding-window chunking.
- Implemented an automated RAG evaluation framework (RAG Triad) to quantify Context Relevance, Faithfulness, and Answer Relevance, reducing hallucination risk.
- Architected multi-provider LLM support (OpenAI, Gemini, Ollama, and offline mocks) and deployed via Docker Compose with an interactive HTML5/ES6 Research Workbench.
```

### 3. Talking Points During an Interview
- **Question**: *"Why did you build your own chunker instead of using default LangChain splitters?"*
- **Answer**: *"Writing the chunker directly allowed me to enforce sentence boundary snapping, control metadata preservation (authors, ArXiv ID, character offsets), and eliminate external framework bloat."*
- **Question**: *"How do you test your AI pipeline in CI/CD?"*
- **Answer**: *"We integrated automated tests in Pytest and configured an offline mock generator mode, enabling the full pipeline to run deterministically and cheaply on every commit."*
